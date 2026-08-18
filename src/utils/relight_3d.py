"""Blender relighting execution and job orchestration for 3D PBR predictions."""

from __future__ import annotations

import contextlib
import json
import subprocess
import tempfile
from collections.abc import Iterator
from pathlib import Path
from typing import Any, Literal

from omegaconf import DictConfig, OmegaConf
from tqdm.auto import tqdm

from src.data.pbr_estimation_dataset_3d import (
    PBREstimationDataset3D,
    PBREstimationSample3D,
)
from src.data.preprocessing.utils import resolve_lights
from src.methods_3d import Prediction3D
from src.utils import get_pylogger

log = get_pylogger(__name__)


@contextlib.contextmanager
def get_relight_3d_working_dir(
    predictions_dir: Path, save_rerenders: bool = False
) -> Iterator[Path]:
    """Provide a permanent or temporary working directory for 3D relit renders."""
    if save_rerenders:
        renders_dir = predictions_dir.parent / "rerenders_3d"
        renders_dir.mkdir(parents=True, exist_ok=True)
        log.info(f"Saving 3D rerenders to {renders_dir}")
        yield renders_dir
    else:
        with tempfile.TemporaryDirectory(prefix="pbr_eval_relight_3d_") as temp_dir:
            yield Path(temp_dir)


def build_relight_job_3d(
    config: DictConfig,
    dataset: PBREstimationDataset3D,
    predictions: dict[str, Prediction3D],
    working_dir: Path,
    *,
    mode: Literal["render", "bake"] = "render",
) -> tuple[dict, dict[str, dict[str, tuple[Path, Path]]], dict[str, str]]:
    """Build a Blender 3D relighting job spec and output path mapping for render or bake mode."""
    targets = resolve_lights(config)
    if config.get("target_envmaps"):
        selected = {str(item) for item in config.target_envmaps}
        targets = [target for target in targets if target["id"] in selected]
        missing = selected - {target["id"] for target in targets}
        if missing:
            raise ValueError(f"Unknown target environment maps: {sorted(missing)}")

    failures: dict[str, str] = {}
    score_paths: dict[str, dict[str, tuple[Path, Path]]] = {}
    sample_jobs: list[dict[str, Any]] = []

    sub_dir = working_dir / mode

    for sample in dataset:
        sid = sample.sample_id
        pred = predictions.get(sid)
        if pred is None:
            failures[sid] = f"Prediction directory missing for {sid}"
            continue

        pred_mesh = pred.pbr_asset_glb
        if pred_mesh is None or not Path(pred_mesh).is_file():
            failures[sid] = f"Missing predicted 3D mesh asset for {sid}"
            continue

        gt_asset = sample.mesh_path
        if not gt_asset.is_file() and sample.asset_path is not None:
            gt_asset = sample.asset_path

        if not gt_asset.is_file():
            failures[sid] = f"Missing ground-truth asset: {sample.mesh_path}"
            continue

        view_tag = (
            sample.reference_view.parent.parent.name
            if sample.reference_view is not None
            else "default"
        )
        outputs = {
            target["id"]: str(
                sub_dir / "pred" / sample.sample_id / f"{target['id']}.png"
            )
            for target in targets
        }
        gt_outputs = (
            {
                target["id"]: str(
                    sub_dir / "gt" / sample.object_id / view_tag / f"{target['id']}.png"
                )
                for target in targets
            }
            if mode == "render"
            else {
                target["id"]: str(
                    sub_dir / "gt" / sample.object_id / f"{target['id']}.png"
                )
                for target in targets
            }
        )

        score_paths[sample.sample_id] = {
            target["id"]: (Path(outputs[target["id"]]), Path(gt_outputs[target["id"]]))
            for target in targets
        }

        camera_data = (
            dict(sample.view_metadata.camera)
            if sample.view_metadata is not None and sample.view_metadata.camera
            else None
        )
        normalization = (
            sample.view_metadata.normalization_source_to_world
            if sample.view_metadata is not None
            else None
        )

        sample_jobs.append(
            {
                "sample_id": sample.sample_id,
                "gt_asset_path": str(gt_asset.resolve()),
                "pred_mesh_path": str(Path(pred_mesh).resolve()),
                "camera": camera_data if mode == "render" else None,
                "normalization": normalization,
                "outputs": outputs,
                "gt_outputs": gt_outputs,
            }
        )

    job_spec = {
        "mode": mode,
        "renderer": OmegaConf.to_container(config.rendering, resolve=True),
        "targets": targets,
        "samples": sample_jobs,
    }
    return job_spec, score_paths, failures


def run_blender_relighting_3d(
    config: DictConfig,
    job_spec: dict,
    total_samples: int,
    working_dir: Path,
    blender_log_path: Path,
    *,
    stage_desc: str = "Indirect 3D PBR relighting (Blender)",
) -> None:
    """Run Blender 3D relighting subprocess with live progress tracking."""
    if not job_spec["samples"]:
        return

    mode = job_spec.get("mode", "render")
    job_path = working_dir / f"job_{mode}.json"
    job_path.write_text(json.dumps(job_spec, indent=2))
    helper = Path(__file__).parent / "_relight_pbr_3d_blender.py"
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
    with open(blender_log_path, "a") as log_file:
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        with tqdm(
            total=total_samples,
            desc=stage_desc,
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


def relight_view_renders_3d(
    config: DictConfig,
    dataset: PBREstimationDataset3D,
    predictions: dict[str, Prediction3D],
    working_dir: Path,
    blender_log_path: Path,
) -> tuple[dict, dict[str, dict[str, tuple[Path, Path]]], dict[str, str]]:
    """Execute view-matched camera relighting pass in Blender."""
    job_spec, score_paths, failures = build_relight_job_3d(
        config, dataset, predictions, working_dir, mode="render"
    )
    log.info(f"Relighting {len(score_paths)} view-matched 3D prediction samples")
    run_blender_relighting_3d(
        config=config,
        job_spec=job_spec,
        total_samples=len(score_paths),
        working_dir=working_dir,
        blender_log_path=blender_log_path,
        stage_desc="Stage 1: View Rerendering (Blender)",
    )
    return job_spec, score_paths, failures


def relight_texture_bakes_3d(
    config: DictConfig,
    dataset: PBREstimationDataset3D,
    predictions: dict[str, Prediction3D],
    working_dir: Path,
    blender_log_path: Path,
) -> tuple[dict, dict[str, dict[str, tuple[Path, Path]]], dict[str, str]]:
    """Execute texture-space UV baking pass in Blender Cycles."""
    job_spec, score_paths, failures = build_relight_job_3d(
        config, dataset, predictions, working_dir, mode="bake"
    )
    log.info(f"Baking textures for {len(score_paths)} 3D prediction samples")
    run_blender_relighting_3d(
        config=config,
        job_spec=job_spec,
        total_samples=len(score_paths),
        working_dir=working_dir,
        blender_log_path=blender_log_path,
        stage_desc="Stage 2: Texture Baking (Blender Cycles)",
    )
    return job_spec, score_paths, failures
