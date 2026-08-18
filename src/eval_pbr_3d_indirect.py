"""Relight predicted 3D GLB assets in Blender and evaluate rendered appearance and baked textures."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import hydra
import rootutils
from hydra.utils import instantiate
from omegaconf import DictConfig
from tqdm.auto import tqdm

PROJECT_ROOT = rootutils.setup_root(
    __file__, indicator=".project_root", pythonpath=True
)

from src.data.pbr_estimation_dataset_3d import PBREstimationDataset3D
from src.methods_3d import Prediction3D
from src.utils.eval import (
    align_resolutions,
    load_alpha,
    load_image,
    load_mask,
    write_yaml,
)
from src.utils.metrics import (
    LPIPSMetric,
    mae,
    mean_metrics,
    psnr,
    rmse,
    ssim,
)
from src.utils.relight_3d import (
    get_relight_3d_working_dir,
    relight_texture_bakes_3d,
    relight_view_renders_3d,
)

log = get_pylogger(__name__)


@dataclass(frozen=True)
class RenderMetrics:
    """Image-space comparison metrics between prediction and reference renders."""

    rmse: float
    psnr: float
    ssim: float
    lpips: float


@dataclass(frozen=True)
class BakeMetrics:
    """Texture-space comparison metrics between baked prediction and reference UV maps."""

    rmse: float
    psnr: float
    mae: float
    ssim: float


def evaluate_relit_render(
    pred_path: Path,
    gt_path: Path,
    lpips_metric: LPIPSMetric,
) -> RenderMetrics:
    """Evaluate image metrics on one relit viewpoint/target pair."""
    prediction = load_image(pred_path, rgb=True)
    target = load_image(gt_path, rgb=True)
    try:
        mask = load_alpha(gt_path)
    except Exception:
        mask = (target > 0.001).any(axis=-1)

    if not mask.any():
        mask = (target > 0.001).any(axis=-1)

    prediction, target, mask = align_resolutions(prediction, target, mask)

    return RenderMetrics(
        rmse=rmse(prediction, target, mask),
        psnr=psnr(prediction, target, mask),
        ssim=ssim(prediction, target, mask),
        lpips=lpips_metric(prediction, target, mask),
    )


def evaluate_baked_texture(
    pred_path: Path,
    gt_path: Path,
    uv_mask_path: Path | None = None,
) -> BakeMetrics:
    """Evaluate texture-space metrics on one baked UV texture pair."""
    prediction = load_image(pred_path, rgb=True)
    target = load_image(gt_path, rgb=True)
    mask = (
        load_mask(uv_mask_path)
        if uv_mask_path is not None and uv_mask_path.is_file()
        else None
    )

    prediction, target, mask = align_resolutions(prediction, target, mask)

    return BakeMetrics(
        rmse=rmse(prediction, target, mask),
        psnr=psnr(prediction, target, mask),
        mae=mae(prediction, target, mask),
        ssim=ssim(prediction, target, mask),
    )


def evaluate_stage_renders(
    dataset: PBREstimationDataset3D,
    score_paths: dict[str, dict[str, tuple[Path, Path]]],
    lpips_metric: LPIPSMetric,
) -> tuple[dict[str, dict], dict[str, Any], dict[str, str]]:
    """Evaluate Stage 1 (view-matched rerendering) across all dataset samples."""
    results: dict[str, dict] = {}
    failures: dict[str, str] = {}
    all_targets: list[dict[str, float]] = []
    relight_targets: list[dict[str, float]] = []
    cycle_targets: list[dict[str, float]] = []

    for sample in tqdm(dataset, desc="Eval View Rerenders", unit="sample"):
        sid = sample.sample_id
        target_paths = score_paths.get(sid)
        if not target_paths:
            continue

        try:
            target_results: dict[str, dict[str, float]] = {}
            for target_id, (pred_render, gt_render) in target_paths.items():
                m = evaluate_relit_render(pred_render, gt_render, lpips_metric)
                m_dict = asdict(m)
                target_results[target_id] = m_dict

                all_targets.append(m_dict)
                if target_id == sample.baked_texture_id:
                    cycle_targets.append(m_dict)
                else:
                    relight_targets.append(m_dict)

            results[sid] = {
                "source": sample.source or "",
                "object_id": sample.object_id,
                "baked_texture_id": sample.baked_texture_id,
                "metrics": mean_metrics(list(target_results.values())),
                "targets": target_results,
            }
        except (FileNotFoundError, ValueError, OSError, TypeError) as error:
            failures[sid] = str(error)

    aggregate = {
        "relight": mean_metrics(relight_targets) if relight_targets else mean_metrics(all_targets),
        "cycle": mean_metrics(cycle_targets) if cycle_targets else None,
        "overall": mean_metrics(all_targets),
    }
    return results, aggregate, failures


def evaluate_stage_bakes(
    dataset: PBREstimationDataset3D,
    score_paths: dict[str, dict[str, tuple[Path, Path]]],
) -> tuple[dict[str, dict], dict[str, Any], dict[str, str]]:
    """Evaluate Stage 2 (texture-space UV baking) across all dataset samples."""
    results: dict[str, dict] = {}
    failures: dict[str, str] = {}
    all_targets: list[dict[str, float]] = []
    relight_targets: list[dict[str, float]] = []
    cycle_targets: list[dict[str, float]] = []

    for sample in tqdm(dataset, desc="Eval Texture Bakes", unit="sample"):
        sid = sample.sample_id
        target_paths = score_paths.get(sid)
        if not target_paths:
            continue

        try:
            target_results: dict[str, dict[str, float]] = {}
            for target_id, (pred_bake, gt_bake) in target_paths.items():
                m = evaluate_baked_texture(pred_bake, gt_bake, sample.uv_mask)
                m_dict = asdict(m)
                target_results[target_id] = m_dict

                all_targets.append(m_dict)
                if target_id == sample.baked_texture_id:
                    cycle_targets.append(m_dict)
                else:
                    relight_targets.append(m_dict)

            results[sid] = {
                "source": sample.source or "",
                "object_id": sample.object_id,
                "baked_texture_id": sample.baked_texture_id,
                "metrics": mean_metrics(list(target_results.values())),
                "targets": target_results,
            }
        except (FileNotFoundError, ValueError, OSError, TypeError) as error:
            failures[sid] = str(error)

    aggregate = {
        "relight": mean_metrics(relight_targets) if relight_targets else mean_metrics(all_targets),
        "cycle": mean_metrics(cycle_targets) if cycle_targets else None,
        "overall": mean_metrics(all_targets),
    }
    return results, aggregate, failures


def evaluate(config: DictConfig) -> dict:
    """Run two-stage 3D indirect evaluation: (1) View Rerendering & (2) Texture Baking."""
    log.info(f"Resolving predictions directory: {config.predictions_dir}")
    predictions_dir = Path(config.predictions_dir).resolve()

    log.info(f"Instantiating 3D dataset <{config.data._target_}>")
    dataset: PBREstimationDataset3D = instantiate(config.data)
    log.info(f"Dataset loaded with {len(dataset)} samples")

    discovered_predictions = (
        {
            d.name: Prediction3D.from_dir(d)
            for d in sorted(predictions_dir.iterdir())
            if d.is_dir()
        }
        if predictions_dir.is_dir()
        else {}
    )
    log.info(f"Discovered {len(discovered_predictions)} prediction directories")

    eval_rerendering = bool(config.get("eval_rerendering", True))
    eval_baking = bool(config.get("eval_baking", True))
    save_rerenders = bool(config.get("save_rerenders", False))
    blender_log_path = predictions_dir.parent / "blender_relight_3d.log"

    payload: dict[str, Any] = {
        "evaluation": "pbr_3d_indirect",
        "predictions_dir": str(predictions_dir),
        "total_requested": len(dataset),
        "eval_rerendering": eval_rerendering,
        "eval_baking": eval_baking,
    }

    all_failures: dict[str, str] = {}

    with get_relight_3d_working_dir(
        predictions_dir, save_rerenders=save_rerenders
    ) as working_dir:
        # --- Stage 1: View Rerendering ---
        if eval_rerendering:
            log.info("Starting Stage 1: View Rerendering Evaluation...")
            job_spec, score_paths, stage1_failures = relight_view_renders_3d(
                config=config,
                dataset=dataset,
                predictions=discovered_predictions,
                working_dir=working_dir,
                blender_log_path=blender_log_path,
            )
            all_failures.update(stage1_failures)

            lpips_metric = LPIPSMetric(
                str(config.device),
                str(config.lpips_backbone),
                Path(config.model_cache_dir).resolve(),
            )
            render_results, render_agg, render_eval_failures = evaluate_stage_renders(
                dataset, score_paths, lpips_metric
            )
            all_failures.update(render_eval_failures)

            payload["target_envmaps"] = [target["id"] for target in job_spec["targets"]]
            payload["view_rerendering"] = {
                "total_evaluated": len(render_results),
                "aggregate": render_agg,
                "samples": render_results,
            }
            log.info(f"Stage 1 (View Rerendering) Aggregate: {render_agg}")

        # --- Stage 2: Texture Baking ---
        if eval_baking:
            log.info("Starting Stage 2: Texture Baking Evaluation...")
            job_spec_bake, bake_score_paths, stage2_failures = relight_texture_bakes_3d(
                config=config,
                dataset=dataset,
                predictions=discovered_predictions,
                working_dir=working_dir,
                blender_log_path=blender_log_path,
            )
            all_failures.update(stage2_failures)

            bake_results, bake_agg, bake_eval_failures = evaluate_stage_bakes(
                dataset, bake_score_paths
            )
            all_failures.update(bake_eval_failures)

            payload["texture_baking"] = {
                "total_evaluated": len(bake_results),
                "aggregate": bake_agg,
                "samples": bake_results,
            }
            log.info(f"Stage 2 (Texture Baking) Aggregate: {bake_agg}")

    payload["failures"] = all_failures

    output_file = (
        Path(config.output_file).resolve()
        if config.get("output_file")
        else predictions_dir.parent / "metrics_indirect.yaml"
    )
    log.info(f"Writing evaluation metrics to {output_file}")
    write_yaml(output_file, payload)

    if all_failures and bool(config.get("strict", False)):
        raise RuntimeError(f"{len(all_failures)} failures out of {len(dataset)} samples")

    return payload


@hydra.main(
    version_base="1.3", config_path="../configs", config_name="eval_pbr_3d_indirect"
)
def main(config: DictConfig) -> None:
    evaluate(config)


if __name__ == "__main__":
    main()
