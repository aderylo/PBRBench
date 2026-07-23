"""Relight screen-space PBR predictions in Blender and compare the renders."""

from __future__ import annotations

import json
import subprocess
import tempfile
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import hydra
import rootutils
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

PROJECT_ROOT = rootutils.setup_root(
    __file__, indicator=".project_root", pythonpath=True
)

from src.data.pbr_estimation_dataset_2d import (  # noqa: E402
    PBREstimationDataset2D,
    PBREstimationSample2D,
)
from src.data.preprocessing.utils import load_subset, resolve_lights  # noqa: E402
from src.utils import get_pylogger  # noqa: E402
from src.utils.eval import (  # noqa: E402
    CHANNELS,
    load_alpha,
    load_image,
    Prediction,
    scan_predictions,
    write_yaml,
)
from src.utils.metrics import (  # noqa: E402
    LPIPSMetric,
    masked_rmse_psnr,
    mean_metrics,
    ssim,
)

log = get_pylogger(__name__)


@dataclass(frozen=True)
class IndirectEvaluationCounts:
    """Counts describing the relation between dataset and prediction artifacts."""

    requested: int
    discovered_predictions: int
    registered_predictions: int
    evaluated: int
    failed: int


@dataclass(frozen=True)
class IndirectSampleResult:
    """Relighting metrics and identifying metadata for one registered sample."""

    object_id: str
    view_id: str
    light_id: str
    metrics: dict[str, float]
    targets: dict[str, dict[str, float]]
    source: str = ""


@dataclass(frozen=True)
class IndirectEvaluationPayload:
    """Complete, YAML-serializable result of an indirect PBR evaluation run."""

    evaluation: str
    predictions_dir: str
    dataset_name: str
    dataset_root: str
    subset_file: str | None
    target_envmaps: list[str]
    counts: IndirectEvaluationCounts
    aggregate: dict[str, float]
    samples: dict[str, IndirectSampleResult]
    failures: dict[str, str]


@dataclass
class RelightingJobState:
    """Paths and validation failures produced while constructing a Blender job."""

    failures: dict[str, str] = field(default_factory=dict)
    score_paths: dict[str, dict[str, tuple[Path, Path]]] = field(
        default_factory=dict
    )


def project_path(value: str | Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def ground_truth_channels(directory: Path) -> dict[str, str]:
    return {name: str((directory / f"{name}.png").resolve()) for name in CHANNELS}


def prediction_channels(prediction: Prediction) -> dict[str, str]:
    return {name: str(path.resolve()) for name, path in prediction.channels.items()}


def build_job(
    config: DictConfig,
    samples: list[PBREstimationSample2D],
    predictions: dict[str, Prediction],
    temporary_dir: Path,
) -> tuple[dict, RelightingJobState]:
    """Build a Blender relighting job from registered, complete predictions."""
    targets = resolve_lights(config)
    if config.target_envmaps:
        selected = {str(item) for item in config.target_envmaps}
        targets = [target for target in targets if target["id"] in selected]
        missing = selected - {target["id"] for target in targets}
        if missing:
            raise ValueError(f"Unknown target environment maps: {sorted(missing)}")

    grouped: dict[str, dict[str, list]] = defaultdict(lambda: defaultdict(list))
    state = RelightingJobState()
    for sample in samples:
        prediction = predictions.get(sample.sample_id)
        if prediction is None:
            state.failures[sample.sample_id] = "Prediction directory missing"
            continue
        missing_channels = prediction.missing_channels()
        if missing_channels:
            state.failures[sample.sample_id] = (
                f"missing prediction channels: {', '.join(missing_channels)}"
            )
            continue
        if any(getattr(sample, name) is None for name in CHANNELS):
            state.failures[sample.sample_id] = "missing ground-truth material channel"
            continue
        sample_targets = [
            target
            for target in targets
            if not bool(config.exclude_source_light) or target["id"] != sample.light_id
        ]
        if not sample_targets:
            state.failures[sample.sample_id] = "no target environment maps remain"
            continue
        grouped[sample.object_id][sample.view_id].append(
            (sample, prediction, sample_targets)
        )

    objects = []
    for object_id, object_views in grouped.items():
        view_jobs = []
        object_metadata = None
        for view_id, entries in object_views.items():
            first = entries[0][0]
            metadata = dict(first.metadata)
            object_metadata = metadata
            view_dir = first.albedo.parent
            target_ids = {
                target["id"]
                for _, _, sample_targets in entries
                for target in sample_targets
            }
            gt_outputs = {
                target_id: str(
                    temporary_dir / "gt" / object_id / view_id / f"{target_id}.png"
                )
                for target_id in target_ids
            }
            prediction_jobs = []
            for sample, prediction, sample_targets in entries:
                outputs = {
                    target["id"]: str(
                        temporary_dir
                        / "pred"
                        / sample.sample_id
                        / f"{target['id']}.png"
                    )
                    for target in sample_targets
                }
                state.score_paths[sample.sample_id] = {
                    target_id: (Path(output), Path(gt_outputs[target_id]))
                    for target_id, output in outputs.items()
                }
                prediction_jobs.append(
                    {
                        "sample_id": sample.sample_id,
                        "channels": prediction_channels(prediction),
                        "outputs": outputs,
                    }
                )
            view_jobs.append(
                {
                    "camera": metadata["camera"],
                    "ground_truth": ground_truth_channels(view_dir),
                    "ground_truth_outputs": gt_outputs,
                    "predictions": prediction_jobs,
                }
            )
        if object_metadata is not None:
            objects.append(
                {
                    "object_id": object_id,
                    "asset_path": object_metadata["asset_path"],
                    "normalization": object_metadata["normalization_source_to_world"],
                    "views": view_jobs,
                }
            )
    return (
        {
            "renderer": OmegaConf.to_container(config.rendering, resolve=True),
            "targets": targets,
            "objects": objects,
        },
        state,
    )


def evaluate(config: DictConfig) -> IndirectEvaluationPayload:
    """Relight registered predictions and evaluate their rendered appearance."""
    predictions_dir = project_path(config.predictions_dir)
    dataset_overrides = {}
    if hasattr(config.data, "root") and config.data.root:
        dataset_overrides["root"] = project_path(config.data.root)
    if config.get("subset_file"):
        dataset_overrides["object_ids"] = load_subset(project_path(config.subset_file))
    log.info("Instantiating dataset <%s>", config.data._target_)
    dataset: PBREstimationDataset2D = instantiate(config.data, **dataset_overrides)
    samples = list(dataset)
    predictions = scan_predictions(predictions_dir, CHANNELS)
    log.info(
        "Found %d prediction directories for %d requested dataset samples",
        len(predictions),
        len(samples),
    )

    with tempfile.TemporaryDirectory(prefix="pbr_eval_relight_") as temp:
        temporary_dir = Path(temp)
        job, state = build_job(config, samples, predictions, temporary_dir)
        log.info("Relighting %d valid prediction samples", len(state.score_paths))
        if job["objects"]:
            job_path = temporary_dir / "job.json"
            job_path.write_text(json.dumps(job, indent=2))
            helper = Path(__file__).parent / "utils" / "_relight_pbr_2d_blender.py"
            subprocess.run(
                [
                    str(config.rendering.executable),
                    "--background",
                    "--python",
                    str(helper),
                    "--",
                    "--job",
                    str(job_path),
                ],
                check=True,
            )

        lpips_metric = LPIPSMetric(
            str(config.device),
            str(config.lpips_backbone),
            project_path(config.model_cache_dir),
        )
        results: dict[str, IndirectSampleResult] = {}
        all_target_metrics: list[dict[str, float]] = []
        sample_lookup = {sample.sample_id: sample for sample in samples}
        for sample_id, target_paths in state.score_paths.items():
            target_results = {}
            try:
                for target_id, (prediction_path, gt_path) in target_paths.items():
                    prediction = load_image(prediction_path, rgb=True)
                    target = load_image(gt_path, rgb=True)
                    mask = load_alpha(gt_path)
                    metrics = masked_rmse_psnr(prediction, target, mask)
                    metrics["ssim"] = ssim(prediction, target, mask)
                    metrics["lpips"] = lpips_metric(prediction, target, mask)
                    target_results[target_id] = metrics
                sample = sample_lookup[sample_id]
                all_target_metrics.extend(target_results.values())
                results[sample_id] = IndirectSampleResult(
                    object_id=sample.object_id,
                    view_id=sample.view_id,
                    light_id=sample.light_id,
                    metrics=mean_metrics(target_results.values()),
                    targets=target_results,
                    source=sample.source,
                )
                log.info("Evaluated %s", sample_id)
            except (FileNotFoundError, ValueError) as error:
                state.failures[sample_id] = str(error)

    payload = IndirectEvaluationPayload(
        evaluation="pbr_2d_indirect",
        predictions_dir=str(predictions_dir),
        dataset_name=dataset.name,
        dataset_root=str(dataset.root),
        subset_file=str(config.subset_file) if config.get("subset_file") else None,
        target_envmaps=[target["id"] for target in job["targets"]],
        counts=IndirectEvaluationCounts(
            requested=len(samples),
            discovered_predictions=len(predictions),
            registered_predictions=sum(
                sample_id in sample_lookup for sample_id in predictions
            ),
            evaluated=len(results),
            failed=len(state.failures),
        ),
        aggregate=mean_metrics(all_target_metrics),
        samples=results,
        failures=state.failures,
    )
    output_file = (
        project_path(config.output_file)
        if config.output_file
        else predictions_dir.parent / "metrics_indirect.yaml"
    )
    write_yaml(output_file, payload)
    log.info(f"Wrote indirect metrics to {output_file}")
    log.info("Aggregate metrics: %s", payload.aggregate)
    if state.failures and bool(config.strict):
        raise RuntimeError(
            f"{len(state.failures)} evaluation failures across {len(samples)} dataset samples"
        )
    return payload


@hydra.main(
    version_base="1.3", config_path="../configs", config_name="eval_pbr_2d_indirect"
)
def main(config: DictConfig) -> None:
    evaluate(config)


if __name__ == "__main__":
    main()
