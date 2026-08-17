"""Evaluate screen-space PBR predictions against registered reference maps."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import hydra
import rootutils
from hydra.utils import instantiate
from omegaconf import DictConfig
from tqdm.auto import tqdm

PROJECT_ROOT = rootutils.setup_root(
    __file__, indicator=".project_root", pythonpath=True
)

from src.data.pbr_estimation_dataset_2d import (  # noqa: E402
    PBREstimationDataset2D,
    PBREstimationSample2D,
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
    masked_rmse_psnr,
    mean_metrics,
    metric_statistics,
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
class DirectSampleResult:
    """Direct metrics and identifying metadata for one registered sample."""

    object_id: str
    view_id: str
    light_id: str
    metrics: dict[str, dict[str, float]]
    source: str = ""


@dataclass(frozen=True)
class DirectMetricSummary:
    """Mean metrics and descriptive statistics for one evaluated group."""

    evaluated: int
    aggregate: dict[str, dict[str, float]]
    statistics: dict[str, dict[str, dict[str, float | int]]]


@dataclass(frozen=True)
class DirectEvaluationPayload:
    """Complete, YAML-serializable result of a direct PBR evaluation run."""

    evaluation: str
    predictions_dir: str
    counts: EvaluationCounts
    aggregate: dict[str, dict[str, float]]
    statistics: dict[str, dict[str, dict[str, float | int]]]
    per_source: dict[str, DirectMetricSummary]
    samples: dict[str, DirectSampleResult]
    failures: dict[str, str]


def project_path(value: str | Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


import numpy as np


def compute_light_consistency_for_group(
    group_samples: list[tuple[PBREstimationSample2D, Prediction]]
) -> dict[str, float]:
    """Compute per-channel light-induced standard deviation across lightings for one object view."""
    if len(group_samples) < 2:
        return {channel: 0.0 for channel in CHANNELS}

    first_sample = group_samples[0][0]
    if first_sample.mask is None or not first_sample.mask.is_file():
        return {channel: 0.0 for channel in CHANNELS}

    mask = load_mask(first_sample.mask)
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
        if mask.any():
            channel_stds[channel] = float(np.mean(std_map[mask]))
        else:
            channel_stds[channel] = float(np.mean(std_map))

    return channel_stds


def evaluate_single_sample(
    sample: PBREstimationSample2D, prediction: Prediction
) -> dict[str, dict[str, float]]:
    """Evaluate direct PBR metrics for a single sample."""
    if sample.mask is None or not sample.mask.is_file():
        raise FileNotFoundError("ground-truth mask is missing")

    mask = load_mask(sample.mask)
    metrics = {}
    for channel in CHANNELS:
        gt_path = getattr(sample, channel)
        pred_path = prediction.channels[channel]
        if gt_path is None or not gt_path.is_file():
            raise FileNotFoundError(f"ground-truth {channel} is missing")
        if not pred_path.is_file():
            raise FileNotFoundError(f"prediction {channel} is missing")

        rgb = channel == "albedo"
        target = load_image(gt_path, rgb=rgb)
        prediction_image = load_image(pred_path, rgb=rgb)

        if rgb:
            target = srgb_to_linear(target)
            prediction_image = srgb_to_linear(prediction_image)

        metrics[channel] = masked_rmse_psnr(prediction_image, target, mask)

    return metrics


def summarize_results(
    results: list[DirectSampleResult],
) -> DirectMetricSummary:
    """Aggregate mean and spread metrics for a set of evaluated samples."""
    sample_metrics = [result.metrics for result in results]
    return DirectMetricSummary(
        evaluated=len(results),
        aggregate=mean_metrics(sample_metrics),
        statistics=metric_statistics(sample_metrics),
    )


def evaluate(config: DictConfig) -> DirectEvaluationPayload:
    """Evaluate prediction artifacts registered in the configured 2D dataset."""
    predictions_dir = project_path(config.predictions_dir)
    log.info("Instantiating dataset <%s>", config.data._target_)
    dataset = instantiate(config.data)
    samples = {sample.sample_id: sample for sample in dataset}
    predictions = scan_predictions(predictions_dir, CHANNELS)

    log.info(
        "Found %d prediction directories for %d requested dataset samples",
        len(predictions),
        len(samples),
    )

    # Group valid predictions by (object_id, view_id) to compute light-induced consistency
    grouped_object_views: dict[tuple[str, str], list[tuple[PBREstimationSample2D, Prediction]]] = defaultdict(list)
    for sample_id in sorted(predictions):
        sample = samples.get(sample_id)
        if sample is not None:
            grouped_object_views[(sample.object_id, sample.view_id)].append(
                (sample, predictions[sample_id])
            )

    consistency_per_view: dict[tuple[str, str], dict[str, float]] = {}
    for obj_view, group_items in grouped_object_views.items():
        consistency_per_view[obj_view] = compute_light_consistency_for_group(group_items)

    results: dict[str, DirectSampleResult] = {}
    failures: dict[str, str] = {}
    for sample_id in tqdm(
        sorted(predictions), desc="Direct PBR evaluation", unit="sample"
    ):
        sample = samples.get(sample_id)
        if sample is None:
            failures[sample_id] = "Prediction directory is not registered in the dataset"
            continue

        try:
            metrics = evaluate_single_sample(sample, predictions[sample_id])
            view_stds = consistency_per_view.get((sample.object_id, sample.view_id), {})
            for channel in CHANNELS:
                metrics[channel]["light_induced_std"] = view_stds.get(channel, 0.0)

            results[sample_id] = DirectSampleResult(
                object_id=sample.object_id,
                view_id=sample.view_id,
                light_id=sample.light_id,
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
    grouped_results: dict[str, list[DirectSampleResult]] = defaultdict(list)
    for result in results.values():
        grouped_results[result.source or "unknown"].append(result)
    per_source = {
        source: summarize_results(source_results)
        for source, source_results in sorted(grouped_results.items())
    }

    payload = DirectEvaluationPayload(
        evaluation="pbr_2d_direct",
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
        # Retain this mean-only field for compatibility with existing reports.
        aggregate=overall.aggregate,
        statistics=overall.statistics,
        per_source=per_source,
        samples=results,
        failures=failures,
    )

    output_file = (
        project_path(config.output_file)
        if config.get("output_file")
        else predictions_dir.parent / "metrics_direct.yaml"
    )
    write_yaml(output_file, payload)
    log.info(f"Wrote direct metrics to {output_file}")
    log.info("Aggregate metrics: %s", payload.aggregate)
    if failures and bool(config.get("strict", False)):
        raise RuntimeError(
            f"{len(failures)} evaluation failures across {len(samples)} dataset samples"
        )
    return payload


@hydra.main(
    version_base="1.3", config_path="../configs", config_name="eval_pbr_2d_direct"
)
def main(config: DictConfig) -> None:
    evaluate(config)


if __name__ == "__main__":
    main()
