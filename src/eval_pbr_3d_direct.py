"""Evaluate 3D PBR predictions against reference ground-truth PBR maps."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import hydra
import numpy as np
import rootutils
from hydra.utils import instantiate
from omegaconf import DictConfig
from PIL import Image
from tqdm.auto import tqdm

PROJECT_ROOT = rootutils.setup_root(
    __file__, indicator=".project_root", pythonpath=True
)

from src.data.pbr_estimation_dataset_3d import (  # noqa: E402
    PBREstimationDataset3D,
    PBREstimationSample3D,
)
from src.utils import get_pylogger  # noqa: E402
from src.utils.eval import (  # noqa: E402
    CHANNELS,
    load_image,
    load_mask,
    Prediction,
    scan_predictions,
    srgb_to_linear,
    write_yaml,
)
from src.utils.metrics import (  # noqa: E402
    mae,
    mean_metrics,
    metric_statistics,
    psnr,
    rmse,
)


log = get_pylogger(__name__)


@dataclass(frozen=True)
class EvaluationCounts:
    """Counts describing the relation between dataset and prediction artifacts."""

    requested: int
    discovered_predictions: int
    registered_predictions: int
    evaluated: int
    failed: int


@dataclass(frozen=True)
class DirectSampleResult3D:
    """Direct metrics and identifying metadata for one registered 3D sample."""

    object_id: str
    texture_id: str
    metrics: dict[str, dict[str, float]]
    source: str = ""


@dataclass(frozen=True)
class DirectMetricSummary:
    """Mean metrics and descriptive statistics for one evaluated group."""

    evaluated: int
    aggregate: dict[str, dict[str, float]]
    statistics: dict[str, dict[str, dict[str, float | int]]]


@dataclass(frozen=True)
class DirectEvaluationPayload3D:
    """Complete, YAML-serializable result of a direct 3D PBR evaluation run."""

    evaluation: str
    predictions_dir: str
    counts: EvaluationCounts
    aggregate: dict[str, dict[str, float]]
    statistics: dict[str, dict[str, dict[str, float | int]]]
    per_source: dict[str, DirectMetricSummary]
    samples: dict[str, DirectSampleResult3D]
    failures: dict[str, str]


def project_path(value: str | Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def find_gt_pbr_channels(sample: PBREstimationSample3D) -> dict[str, Path]:
    """Locate reference ground truth PBR map paths (albedo, roughness, metallic) for a 3D sample."""
    obj_dir = sample.mesh_path.parent
    pbr_dir = obj_dir / "pbr" if (obj_dir / "pbr").is_dir() else obj_dir

    gt_paths: dict[str, Path] = {}
    for channel in CHANNELS:
        candidates = [
            pbr_dir / f"{channel}.png",
            pbr_dir / f"{channel}.jpg",
            obj_dir / f"{channel}.png",
            obj_dir / f"{channel}.jpg",
        ]
        for cand in candidates:
            if cand.is_file():
                gt_paths[channel] = cand
                break
    return gt_paths


def find_gt_mask(sample: PBREstimationSample3D, gt_paths: dict[str, Path]) -> Path | None:
    """Locate optional ground truth mask image file for a 3D sample."""
    obj_dir = sample.mesh_path.parent
    candidates = [
        obj_dir / "uv_mask.png",
        obj_dir / "pbr" / "uv_mask.png",
        obj_dir / "mask.png",
        obj_dir / "pbr" / "mask.png",
    ]
    for cand in candidates:
        if cand.is_file():
            return cand
    return None


def get_mask_from_image_or_alpha(image_path: Path, target_shape: tuple[int, int]) -> np.ndarray:
    """Extract alpha mask or foreground mask from image, or fallback to full mask."""
    try:
        with Image.open(image_path) as img:
            if "A" in img.getbands():
                alpha = np.asarray(img.getchannel("A")) > 127
                if alpha.shape == target_shape and alpha.any():
                    return alpha
            arr = np.asarray(img.convert("L"))
            mask = arr > 0
            if mask.shape == target_shape and mask.any():
                return mask
    except Exception:
        pass
    return np.ones(target_shape, dtype=bool)


def compute_light_consistency_for_group(
    group_samples: list[tuple[PBREstimationSample3D, Prediction]]
) -> dict[str, float]:
    """Compute per-channel light-induced standard deviation across lightings/textures for one 3D object."""
    if len(group_samples) < 2:
        return {channel: 0.0 for channel in CHANNELS}

    first_sample = group_samples[0][0]
    gt_paths = find_gt_pbr_channels(first_sample)
    mask_path = find_gt_mask(first_sample, gt_paths)

    channel_stds = {}
    for channel in CHANNELS:
        rgb = channel == "albedo"
        preds = []
        for sample, prediction in group_samples:
            pred_path = prediction.channels[channel]
            if not pred_path.is_file():
                continue
            pred_img = load_image(pred_path, rgb=rgb)
            if rgb:
                pred_img = srgb_to_linear(pred_img)
            preds.append(pred_img)

        if len(preds) < 2:
            channel_stds[channel] = 0.0
            continue

        preds_array = np.stack(preds, axis=0)
        std_map = np.std(preds_array, axis=0)

        target_shape = preds_array.shape[1:3]
        if mask_path is not None and mask_path.is_file():
            mask = load_mask(mask_path)
            if mask.shape != target_shape:
                mask_pil = Image.open(mask_path).convert("L").resize(
                    (target_shape[1], target_shape[0]), Image.Resampling.NEAREST
                )
                mask = np.asarray(mask_pil) > 0
        elif channel in gt_paths and gt_paths[channel].is_file():
            mask = get_mask_from_image_or_alpha(gt_paths[channel], target_shape)
        else:
            mask = np.ones(target_shape, dtype=bool)

        if mask.any():
            channel_stds[channel] = float(np.mean(std_map[mask]))
        else:
            channel_stds[channel] = float(np.mean(std_map))

    return channel_stds


def evaluate_single_sample(
    sample: PBREstimationSample3D, prediction: Prediction
) -> dict[str, dict[str, float]]:
    """Evaluate direct PBR metrics (MAE, RMSE, PSNR) for a single 3D sample."""
    gt_paths = find_gt_pbr_channels(sample)
    mask_path = find_gt_mask(sample, gt_paths)

    metrics = {}
    for channel in CHANNELS:
        gt_path = gt_paths.get(channel)
        pred_path = prediction.channels[channel]

        if gt_path is None or not gt_path.is_file():
            raise FileNotFoundError(f"ground-truth {channel} map is missing for sample {sample.sample_id}")
        if not pred_path.is_file():
            raise FileNotFoundError(f"prediction {channel} map is missing for sample {sample.sample_id}")

        rgb = channel == "albedo"
        target = load_image(gt_path, rgb=rgb)
        prediction_image = load_image(pred_path, rgb=rgb)

        if target.shape[:2] != prediction_image.shape[:2]:
            pred_pil = Image.open(pred_path).convert("RGB" if rgb else "L").resize(
                (target.shape[1], target.shape[0]), Image.Resampling.BILINEAR
            )
            prediction_image = np.asarray(pred_pil, dtype=np.float32) / 255.0

        if mask_path is not None and mask_path.is_file():
            mask = load_mask(mask_path)
            if mask.shape != target.shape[:2]:
                mask_pil = Image.open(mask_path).convert("L").resize(
                    (target.shape[1], target.shape[0]), Image.Resampling.NEAREST
                )
                mask = np.asarray(mask_pil) > 0
        else:
            mask = get_mask_from_image_or_alpha(gt_path, target.shape[:2])

        if rgb:
            target = srgb_to_linear(target)
            prediction_image = srgb_to_linear(prediction_image)

        metrics[channel] = {
            "mae": mae(prediction_image, target, mask),
            "rmse": rmse(prediction_image, target, mask),
            "psnr": psnr(prediction_image, target, mask),
        }

    return metrics



def summarize_results(
    results: list[DirectSampleResult3D],
) -> DirectMetricSummary:
    """Aggregate mean and spread metrics for a set of evaluated 3D samples."""
    sample_metrics = [result.metrics for result in results]
    return DirectMetricSummary(
        evaluated=len(results),
        aggregate=mean_metrics(sample_metrics),
        statistics=metric_statistics(sample_metrics),
    )


def evaluate(config: DictConfig) -> DirectEvaluationPayload3D:
    """Evaluate 3D prediction artifacts registered in the configured 3D dataset."""
    predictions_dir = project_path(config.predictions_dir)
    log.info("Instantiating dataset <%s>", config.data._target_)
    dataset = instantiate(config.data)
    samples = {sample.sample_id: sample for sample in dataset}
    predictions = scan_predictions(predictions_dir, CHANNELS)

    log.info(
        "Found %d prediction directories for %d requested 3D dataset samples",
        len(predictions),
        len(samples),
    )

    # Group valid predictions by object_id to compute light/texture consistency
    grouped_objects: dict[str, list[tuple[PBREstimationSample3D, Prediction]]] = defaultdict(list)
    for sample_id in sorted(predictions):
        sample = samples.get(sample_id)
        if sample is not None:
            grouped_objects[sample.object_id].append((sample, predictions[sample_id]))

    consistency_per_object: dict[str, dict[str, float]] = {}
    for object_id, group_items in grouped_objects.items():
        consistency_per_object[object_id] = compute_light_consistency_for_group(group_items)

    results: dict[str, DirectSampleResult3D] = {}
    failures: dict[str, str] = {}
    for sample_id in tqdm(
        sorted(predictions), desc="Direct 3D PBR evaluation", unit="sample"
    ):
        sample = samples.get(sample_id)
        if sample is None:
            failures[sample_id] = "Prediction directory is not registered in the dataset"
            continue

        try:
            metrics = evaluate_single_sample(sample, predictions[sample_id])
            obj_stds = consistency_per_object.get(sample.object_id, {})
            for channel in CHANNELS:
                metrics[channel]["light_induced_std"] = obj_stds.get(channel, 0.0)

            results[sample_id] = DirectSampleResult3D(
                object_id=sample.object_id,
                texture_id=sample.texture_id,
                metrics=metrics,
                source=sample.source,
            )
        except (FileNotFoundError, ValueError) as error:
            failures[sample_id] = str(error)

    for sample_id in sorted(samples.keys() - predictions.keys()):
        failures[sample_id] = (
            "Prediction directory missing: " f"{predictions_dir / sample_id}"
        )

    overall = summarize_results(list(results.values()))
    grouped_results: dict[str, list[DirectSampleResult3D]] = defaultdict(list)
    for result in results.values():
        grouped_results[result.source or "unknown"].append(result)
    per_source = {
        source: summarize_results(source_results)
        for source, source_results in sorted(grouped_results.items())
    }

    payload = DirectEvaluationPayload3D(
        evaluation="pbr_3d_direct",
        predictions_dir=str(predictions_dir),
        counts=EvaluationCounts(
            requested=len(samples),
            discovered_predictions=len(predictions),
            registered_predictions=sum(
                sample_id in samples for sample_id in predictions
            ),
            evaluated=len(results),
            failed=len(failures),
        ),
        aggregate=overall.aggregate,
        statistics=overall.statistics,
        per_source=per_source,
        samples=results,
        failures=failures,
    )

    output_file = (
        project_path(config.output_file)
        if config.get("output_file")
        else predictions_dir.parent / "direct_metrics.yaml"
    )
    write_yaml(output_file, payload)
    log.info(f"Wrote direct 3D metrics to {output_file}")
    log.info("Aggregate metrics: %s", payload.aggregate)
    if failures and bool(config.get("strict", False)):
        raise RuntimeError(
            f"{len(failures)} evaluation failures across {len(samples)} 3D dataset samples"
        )
    return payload


@hydra.main(
    version_base="1.3", config_path="../configs", config_name="eval_pbr_3d_direct"
)
def main(config: DictConfig) -> None:
    evaluate(config)


if __name__ == "__main__":
    main()
