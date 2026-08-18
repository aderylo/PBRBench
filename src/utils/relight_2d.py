"""Blender relighting execution and job orchestration for 2D screen-space PBR predictions."""

from __future__ import annotations

import contextlib
import json
import subprocess
import tempfile
from collections import defaultdict
from collections.abc import Iterator
from pathlib import Path

from omegaconf import DictConfig, OmegaConf
from tqdm.auto import tqdm

from src.data.pbr_estimation_dataset_2d import (
    PBREstimationDataset2D,
    PBREstimationSample2D,
)
from src.data.preprocessing.utils import resolve_lights
from src.methods_2d import Prediction2D
from src.utils import get_pylogger

log = get_pylogger(__name__)


@contextlib.contextmanager
def get_relight_working_dir(
    predictions_dir: Path, save_rerenders: bool = False
) -> Iterator[Path]:
    """Provide a permanent or temporary working directory for relit renders."""
    if save_rerenders:
        renders_dir = predictions_dir.parent / "rerenders"
        renders_dir.mkdir(parents=True, exist_ok=True)
        log.info(f"Saving rerenders to {renders_dir}")
        yield renders_dir
    else:
        with tempfile.TemporaryDirectory(prefix="pbr_eval_relight_") as temp_dir:
            yield Path(temp_dir)


def build_relight_job_2d(
    config: DictConfig,
    dataset: PBREstimationDataset2D,
    predictions: dict[str, Prediction2D],
    working_dir: Path,
) -> tuple[dict, dict[str, dict[str, tuple[Path, Path]]], dict[str, str]]:
    """Build a Blender relighting job spec and output path mapping from registered predictions."""
    targets = resolve_lights(config)
    if config.target_envmaps:
        selected = {str(item) for item in config.target_envmaps}
        targets = [target for target in targets if target["id"] in selected]
        missing = selected - {target["id"] for target in targets}
        if missing:
            raise ValueError(f"Unknown target environment maps: {sorted(missing)}")

    failures: dict[str, str] = {}
    score_paths: dict[str, dict[str, tuple[Path, Path]]] = {}
    grouped: dict[str, dict[str, list[tuple[PBREstimationSample2D, Prediction2D, list]]]] = (
        defaultdict(lambda: defaultdict(list))
    )

    for sample in dataset:
        sid = sample.sample_id
        pred = predictions.get(sid)
        if pred is None:
            failures[sid] = f"Prediction directory missing for {sid}"
            continue

        pred_albedo = Path(pred.albedo)
        pred_roughness = Path(pred.roughness)
        pred_metallic = Path(pred.metallic)
        if not (pred_albedo.is_file() and pred_roughness.is_file() and pred_metallic.is_file()):
            missing = [
                name
                for name, p in [
                    ("albedo", pred_albedo),
                    ("roughness", pred_roughness),
                    ("metallic", pred_metallic),
                ]
                if not p.is_file()
            ]
            failures[sid] = f"Missing prediction channels: {', '.join(missing)}"
            continue

        if not (sample.albedo and sample.roughness and sample.metallic):
            failures[sid] = "Missing ground-truth material channels"
            continue

        grouped[sample.object_id][sample.view_id].append((sample, pred, targets))

    objects = []
    for object_id, object_views in grouped.items():
        view_jobs = []
        object_metadata = None
        for view_id, entries in object_views.items():
            first_sample = entries[0][0]
            metadata = dict(first_sample.metadata)
            object_metadata = metadata

            gt_channels = {
                "albedo": str(first_sample.albedo.resolve()),
                "roughness": str(first_sample.roughness.resolve()),
                "metallic": str(first_sample.metallic.resolve()),
            }

            target_ids = {
                target["id"]
                for _, _, sample_targets in entries
                for target in sample_targets
            }
            gt_outputs = {
                target_id: str(
                    working_dir / "gt" / object_id / view_id / f"{target_id}.png"
                )
                for target_id in target_ids
            }

            prediction_jobs = []
            for sample, pred, sample_targets in entries:
                outputs = {
                    target["id"]: str(
                        working_dir
                        / "pred"
                        / sample.sample_id
                        / f"{target['id']}.png"
                    )
                    for target in sample_targets
                }
                score_paths[sample.sample_id] = {
                    target_id: (Path(outputs[target_id]), Path(gt_outputs[target_id]))
                    for target_id in outputs
                }
                prediction_jobs.append(
                    {
                        "sample_id": sample.sample_id,
                        "channels": {
                            "albedo": str(Path(pred.albedo).resolve()),
                            "roughness": str(Path(pred.roughness).resolve()),
                            "metallic": str(Path(pred.metallic).resolve()),
                        },
                        "outputs": outputs,
                    }
                )

            view_jobs.append(
                {
                    "camera": metadata["camera"],
                    "ground_truth": gt_channels,
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

    job_spec = {
        "renderer": OmegaConf.to_container(config.rendering, resolve=True),
        "targets": targets,
        "objects": objects,
    }
    return job_spec, score_paths, failures


def run_blender_relighting_2d(
    config: DictConfig,
    job_spec: dict,
    total_samples: int,
    working_dir: Path,
    blender_log_path: Path,
) -> None:
    """Run Blender projective relighting subprocess with live progress tracking."""
    if not job_spec["objects"]:
        return

    job_path = working_dir / "job.json"
    job_path.write_text(json.dumps(job_spec, indent=2))
    helper = Path(__file__).parent / "_relight_pbr_2d_blender.py"
    log.info(f"Blender log saved to {blender_log_path}")

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
            total=total_samples,
            desc="Indirect PBR relighting (Blender)",
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


def relight_dataset_2d(
    config: DictConfig,
    dataset: PBREstimationDataset2D,
    predictions: dict[str, Prediction2D],
    working_dir: Path,
    blender_log_path: Path,
) -> tuple[dict, dict[str, dict[str, tuple[Path, Path]]], dict[str, str]]:
    """Orchestrate full 2D relighting in Blender, returning (job_spec, score_paths, failures)."""
    job_spec, score_paths, failures = build_relight_job_2d(
        config, dataset, predictions, working_dir
    )
    log.info(f"Relighting {len(score_paths)} valid prediction samples")

    run_blender_relighting_2d(
        config=config,
        job_spec=job_spec,
        total_samples=len(score_paths),
        working_dir=working_dir,
        blender_log_path=blender_log_path,
    )

    return job_spec, score_paths, failures
