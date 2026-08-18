"""Relight predicted 3D GLB assets in Blender and evaluate rendered appearance."""

from __future__ import annotations

import contextlib
import json
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

import hydra
import rootutils
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from tqdm.auto import tqdm

PROJECT_ROOT = rootutils.setup_root(
    __file__, indicator=".project_root", pythonpath=True
)

from src.data.pbr_estimation_dataset_3d import (  # noqa: E402
    PBREstimationDataset3D,
    PBREstimationSample3D,
)
from src.data.preprocessing.utils import resolve_lights  # noqa: E402
from src.utils import get_pylogger  # noqa: E402
from src.utils.eval import (  # noqa: E402
    load_alpha,
    load_image,
    write_yaml,
)
from src.utils.metrics import (  # noqa: E402
    LPIPSMetric,
    mean_metrics,
    psnr,
    rmse,
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
class IndirectSampleResult3D:
    """Indirect metrics and identifying metadata for one registered 3D sample."""

    object_id: str
    texture_id: str
    metrics: dict[str, float]
    targets: dict[str, dict[str, float]]
    source: str = ""


@dataclass(frozen=True)
class IndirectEvaluationPayload3D:
    """Complete, YAML-serializable result of an indirect 3D PBR evaluation run."""

    evaluation: str
    predictions_dir: str
    target_envmaps: list[str]
    counts: IndirectEvaluationCounts
    aggregate: dict[str, Any]
    samples: dict[str, IndirectSampleResult3D]
    failures: dict[str, str]


@dataclass
class RelightingJobState3D:
    """Paths and validation failures produced while constructing a Blender 3D relighting job."""

    failures: dict[str, str] = field(default_factory=dict)
    score_paths: dict[str, dict[str, tuple[Path, Path]]] = field(
        default_factory=dict
    )


def project_path(value: str | Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def build_job(
    config: DictConfig,
    samples: list[PBREstimationSample3D],
    predictions_dir: Path,
    temporary_dir: Path,
) -> tuple[dict, RelightingJobState3D]:
    """Build a Blender relighting job for predicted 3D GLB assets."""
    targets = resolve_lights(config)
    if config.target_envmaps:
        selected = {str(item) for item in config.target_envmaps}
        targets = [target for target in targets if target["id"] in selected]
        missing = selected - {target["id"] for target in targets}
        if missing:
            raise ValueError(f"Unknown target environment maps: {sorted(missing)}")

    state = RelightingJobState3D()
    sample_jobs = []

    for sample in samples:
        sample_dir = predictions_dir / sample.sample_id
        if not sample_dir.is_dir():
            state.failures[sample.sample_id] = f"Prediction directory missing: {sample_dir}"
            continue

        pred_mesh = sample_dir / "mesh.glb"
        if not pred_mesh.is_file():
            for alt in ("mesh.gltf", "mesh.obj"):
                if (sample_dir / alt).is_file():
                    pred_mesh = sample_dir / alt
                    break

        if not pred_mesh.is_file():
            state.failures[sample.sample_id] = "missing predicted 3D mesh (mesh.glb/mesh.obj)"
            continue

        gt_asset = sample.mesh_path
        if not gt_asset.is_file() and "asset_path" in sample.metadata:
            gt_asset = Path(sample.metadata["asset_path"])

        outputs = {
            target["id"]: str(
                temporary_dir / "pred" / sample.sample_id / f"{target['id']}.png"
            )
            for target in targets
        }
        gt_outputs = {
            target["id"]: str(
                temporary_dir / "gt" / sample.object_id / f"{target['id']}.png"
            )
            for target in targets
        }

        state.score_paths[sample.sample_id] = {
            target["id"]: (Path(outputs[target["id"]]), Path(gt_outputs[target["id"]]))
            for target in targets
        }

        sample_jobs.append(
            {
                "sample_id": sample.sample_id,
                "gt_asset_path": str(gt_asset.resolve()),
                "pred_mesh_path": str(pred_mesh.resolve()),
                "outputs": outputs,
                "gt_outputs": gt_outputs,
            }
        )

    return (
        {
            "renderer": OmegaConf.to_container(config.rendering, resolve=True),
            "targets": targets,
            "samples": sample_jobs,
        },
        state,
    )


def evaluate(config: DictConfig) -> IndirectEvaluationPayload3D:
    """Relight predicted 3D GLB assets and evaluate rendered appearance metrics."""
    predictions_dir = project_path(config.predictions_dir)
    log.info("Instantiating dataset <%s>", config.data._target_)
    dataset: PBREstimationDataset3D = instantiate(config.data)
    samples = list(dataset)
    discovered_predictions = len([d for d in predictions_dir.iterdir() if d.is_dir()]) if predictions_dir.is_dir() else 0
    log.info(
        "Found %d prediction directories for %d requested 3D dataset samples",
        discovered_predictions,
        len(samples),
    )

    should_save_renders = bool(config.get("save_rerenders"))
    if should_save_renders:
        renders_dir = predictions_dir.parent / "rerenders_3d"
        renders_dir.mkdir(parents=True, exist_ok=True)
        cm = contextlib.nullcontext(renders_dir)
        log.info("Saving 3D rerenders to %s", renders_dir)
    else:
        renders_dir = None
        cm = tempfile.TemporaryDirectory(prefix="pbr_eval_relight_3d_")

    with cm as target_dir_raw:
        working_dir = Path(target_dir_raw)
        job, state = build_job(config, samples, predictions_dir, working_dir)
        log.info("Relighting %d valid 3D prediction samples", len(state.score_paths))
        if job["samples"]:
            job_path = working_dir / "job.json"
            job_path.write_text(json.dumps(job, indent=2))
            helper = Path(__file__).parent / "utils" / "_relight_pbr_3d_blender.py"
            blender_log_path = predictions_dir.parent / "blender_relight_3d.log"
            log.info("Blender log saved to %s", blender_log_path)

            cmd = [
                str(config.rendering.executable),
                "--background",
                "--python",
                str(helper),
                "--",
                "--job",
                str(job_path),
            ]
            with open(blender_log_path, "w") as log_file:
                process = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1,
                )
                with tqdm(
                    total=len(state.score_paths),
                    desc="Indirect 3D PBR relighting (Blender)",
                    unit="sample",
                ) as pbar:
                    if process.stdout:
                        for line in process.stdout:
                            log_file.write(line)
                            if line.startswith("PROGRESS "):
                                pbar.update(1)
                return_code = process.wait()
                if return_code != 0:
                    raise subprocess.CalledProcessError(return_code, cmd)

        lpips_metric = LPIPSMetric(
            str(config.device),
            str(config.lpips_backbone),
            project_path(config.model_cache_dir),
        )
        results: dict[str, IndirectSampleResult3D] = {}
        all_target_metrics: list[dict[str, float]] = []
        relight_target_metrics: list[dict[str, float]] = []
        cycle_target_metrics: list[dict[str, float]] = []
        sample_lookup = {sample.sample_id: sample for sample in samples}

        for sample_id, target_paths in state.score_paths.items():
            target_results = {}
            sample = sample_lookup.get(sample_id)
            if sample is None:
                continue
            try:
                for target_id, (prediction_path, gt_path) in target_paths.items():
                    prediction = load_image(prediction_path, rgb=True)
                    target = load_image(gt_path, rgb=True)
                    try:
                        mask = load_alpha(gt_path)
                    except Exception:
                        mask = (target > 0.001).any(axis=-1)

                    if not mask.any():
                        mask = (target > 0.001).any(axis=-1)
                    metrics = {
                        "rmse": rmse(prediction, target, mask),
                        "psnr": psnr(prediction, target, mask),
                    }
                    metrics["ssim"] = ssim(prediction, target, mask)
                    metrics["lpips"] = lpips_metric(prediction, target, mask)
                    target_results[target_id] = metrics


                    all_target_metrics.append(metrics)
                    if target_id == sample.texture_id:
                        cycle_target_metrics.append(metrics)
                    else:
                        relight_target_metrics.append(metrics)

                results[sample_id] = IndirectSampleResult3D(
                    object_id=sample.object_id,
                    texture_id=sample.texture_id,
                    metrics=mean_metrics(target_results.values()),
                    targets=target_results,
                    source=sample.source,
                )
                log.info("Evaluated 3D indirect relighting for %s", sample_id)
            except (FileNotFoundError, ValueError) as error:
                state.failures[sample_id] = str(error)

    aggregate = {
        "relight": mean_metrics(relight_target_metrics) if relight_target_metrics else mean_metrics(all_target_metrics),
        "cycle": mean_metrics(cycle_target_metrics) if cycle_target_metrics else None,
        "overall": mean_metrics(all_target_metrics),
    }

    payload = IndirectEvaluationPayload3D(
        evaluation="pbr_3d_indirect",
        predictions_dir=str(predictions_dir),
        target_envmaps=[target["id"] for target in job["targets"]],
        counts=IndirectEvaluationCounts(
            requested=len(samples),
            discovered_predictions=discovered_predictions,
            registered_predictions=sum(
                sample_id in sample_lookup for sample_id in state.score_paths
            ),
            evaluated=len(results),
            failed=len(state.failures),
        ),
        aggregate=aggregate,
        samples=results,
        failures=state.failures,
    )

    output_file = (
        project_path(config.output_file)
        if config.get("output_file")
        else predictions_dir.parent / "indirect_metrics.yaml"
    )
    write_yaml(output_file, payload)
    log.info(f"Wrote indirect 3D metrics to {output_file}")
    log.info("Aggregate metrics: %s", payload.aggregate)
    if state.failures and bool(config.get("strict", False)):
        raise RuntimeError(
            f"{len(state.failures)} evaluation failures across {len(samples)} 3D dataset samples"
        )
    return payload


@hydra.main(
    version_base="1.3", config_path="../configs", config_name="eval_pbr_3d_indirect"
)
def main(config: DictConfig) -> None:
    evaluate(config)


if __name__ == "__main__":
    main()
