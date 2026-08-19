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

from src.data.envmaps import EnvMapDataset
from src.data.pbr_estimation_dataset_3d import PBREstimationDataset3D
from src.utils import get_pylogger
from src.utils.eval import (
    RerenderingImageEvaluator,
    RerenderingUVEvaluator,
    gt_sample3d_to_render_item,
    pred3d_to_render_item,
    scan_pbr_predictions_dir_3d,
    write_yaml,
)
from src.utils.metrics import (
    mean_metrics,
)
from src.utils.rerender_3d_orchestrator import (
    Rerenderer3D,
)

log = get_pylogger(__name__)


def evaluate(config: DictConfig) -> dict:
    """Run two-stage 3D indirect evaluation: (1) View Rerendering & (2) Texture Baking."""
    log.info(f"Resolving predictions directory: {config.predictions_dir}")
    predictions_dir = Path(config.predictions_dir).resolve()

    log.info(f"Instantiating 3D dataset <{config.data._target_}>")
    dataset: PBREstimationDataset3D = instantiate(config.data)
    log.info(f"Dataset loaded with {len(dataset)} samples")

    log.info(f"Scanning predictions in {predictions_dir}")
    predictions = scan_pbr_predictions_dir_3d(predictions_dir)
    log.info(f"Discovered {len(predictions)} prediction directories")

    log.info(f"Instantiating lighting dataset <{config.lighting._target_}>")
    envmaps: EnvMapDataset = instantiate(config.lighting)

    matched_samples = [
        (sample, predictions[sample.sample_id])
        for sample in dataset
        if sample.sample_id in predictions
        and predictions[sample.sample_id].pbr_asset_glb is not None
        and Path(predictions[sample.sample_id].pbr_asset_glb).is_file()
    ]
    log.info(f"Matched {len(matched_samples)}/{len(dataset)} valid predictions with GT")

    eval_rerendering = bool(config.get("eval_rerendering", True))
    eval_baking = bool(config.get("eval_baking", True))

    rerenders_dir = predictions_dir.parent / "rerenders"
    blender_log_path = predictions_dir.parent / "blender_relight_3d.log"

    log.info(f"Instantiating rerenderer <{config.rerenderer._target_}>")
    rerenderer: Rerenderer3D = instantiate(config.rerenderer)

    image_evaluator: RerenderingImageEvaluator | None = None
    if eval_rerendering:
        log.info(f"Instantiating image evaluator <{config.image_evaluator._target_}>")
        image_evaluator = instantiate(config.image_evaluator)

    uv_evaluator: RerenderingUVEvaluator | None = None
    if eval_baking:
        log.info(f"Instantiating UV evaluator <{config.uv_evaluator._target_}>")
        uv_evaluator = instantiate(config.uv_evaluator)

    payload: dict[str, Any] = {
        "evaluation": "pbr_3d_indirect",
        "predictions_dir": str(predictions_dir),
        "target_envmaps": [e.id for e in envmaps],
        "total_requested": len(dataset),
        "eval_rerendering": eval_rerendering,
        "eval_baking": eval_baking,
    }

    all_failures: dict[str, str] = {}

    # --- Stage 1: View Rerendering ---
    if eval_rerendering:
        log.info("Starting Stage 1: View Rerendering...")
        gt_render_items = [
            gt_sample3d_to_render_item(sample, envmap, rerenders_dir, mode="render")
            for sample, _ in matched_samples
            for envmap in envmaps
        ]
        pred_render_items = [
            pred3d_to_render_item(pred, sample, envmap, rerenders_dir, mode="render")
            for sample, pred in matched_samples
            for envmap in envmaps
        ]

        rerenderer.render(
            items=gt_render_items + pred_render_items,
            working_dir=rerenders_dir,
            blender_log_path=blender_log_path,
            desc="Stage 1: View Rerendering (Blender)",
        )

        stage1_results: dict[str, dict] = {}
        all_render_metrics: list[dict[str, float]] = []
        relight_render_metrics: list[dict[str, float]] = []
        cycle_render_metrics: list[dict[str, float]] = []

        for (sample, pred) in tqdm(
            matched_samples,
            desc="Eval View Rerenders",
            unit="sample",
        ):
            sid = sample.sample_id
            target_results: dict[str, dict[str, float]] = {}
            has_error = False

            for envmap in envmaps:
                pred_item = pred3d_to_render_item(pred, sample, envmap, rerenders_dir, mode="render")
                gt_item = gt_sample3d_to_render_item(sample, envmap, rerenders_dir, mode="render")
                try:
                    m = image_evaluator.evaluate(pred_item.output_path, gt_item.output_path)
                    target_results[envmap.id] = m
                    all_render_metrics.append(m)
                    if envmap.id == sample.baked_texture_id:
                        cycle_render_metrics.append(m)
                    else:
                        relight_render_metrics.append(m)
                except (FileNotFoundError, ValueError, OSError, TypeError) as error:
                    all_failures[f"{sid}__{envmap.id}__render"] = str(error)
                    has_error = True

            if target_results and not has_error:
                stage1_results[sid] = {
                    "source": sample.source or "",
                    "object_id": sample.object_id,
                    "baked_texture_id": sample.baked_texture_id,
                    "metrics": mean_metrics(list(target_results.values())),
                    "targets": target_results,
                }

        render_agg = {
            "relight": mean_metrics(relight_render_metrics) if relight_render_metrics else mean_metrics(all_render_metrics),
            "cycle": mean_metrics(cycle_render_metrics) if cycle_render_metrics else None,
            "overall": mean_metrics(all_render_metrics),
        }
        payload["view_rerendering"] = {
            "total_evaluated": len(stage1_results),
            "aggregate": render_agg,
            "samples": stage1_results,
        }
        log.info(f"Stage 1 (View Rerendering) Aggregate: {render_agg}")

    # --- Stage 2: Texture Baking ---
    if eval_baking:
        log.info("Starting Stage 2: Texture Baking...")
        gt_bake_items = [
            gt_sample3d_to_render_item(sample, envmap, rerenders_dir, mode="bake")
            for sample, _ in matched_samples
            for envmap in envmaps
        ]
        pred_bake_items = [
            pred3d_to_render_item(pred, sample, envmap, rerenders_dir, mode="bake")
            for sample, pred in matched_samples
            for envmap in envmaps
        ]

        rerenderer.render(
            items=gt_bake_items + pred_bake_items,
            working_dir=rerenders_dir,
            blender_log_path=blender_log_path,
            desc="Stage 2: Texture Baking (Blender Cycles)",
        )

        stage2_results: dict[str, dict] = {}
        all_bake_metrics: list[dict[str, float]] = []
        relight_bake_metrics: list[dict[str, float]] = []
        cycle_bake_metrics: list[dict[str, float]] = []

        for (sample, pred) in tqdm(
            matched_samples,
            desc="Eval Texture Bakes",
            unit="sample",
        ):
            sid = sample.sample_id
            target_results: dict[str, dict[str, float]] = {}
            has_error = False

            for envmap in envmaps:
                pred_item = pred3d_to_render_item(pred, sample, envmap, rerenders_dir, mode="bake")
                gt_item = gt_sample3d_to_render_item(sample, envmap, rerenders_dir, mode="bake")
                try:
                    m = uv_evaluator.evaluate(pred_item.output_path, gt_item.output_path, sample.uv_mask)
                    target_results[envmap.id] = m
                    all_bake_metrics.append(m)
                    if envmap.id == sample.baked_texture_id:
                        cycle_bake_metrics.append(m)
                    else:
                        relight_bake_metrics.append(m)
                except (FileNotFoundError, ValueError, OSError, TypeError) as error:
                    all_failures[f"{sid}__{envmap.id}__bake"] = str(error)
                    has_error = True

            if target_results and not has_error:
                stage2_results[sid] = {
                    "source": sample.source or "",
                    "object_id": sample.object_id,
                    "baked_texture_id": sample.baked_texture_id,
                    "metrics": mean_metrics(list(target_results.values())),
                    "targets": target_results,
                }

        bake_agg = {
            "relight": mean_metrics(relight_bake_metrics) if relight_bake_metrics else mean_metrics(all_bake_metrics),
            "cycle": mean_metrics(cycle_bake_metrics) if cycle_bake_metrics else None,
            "overall": mean_metrics(all_bake_metrics),
        }
        payload["texture_baking"] = {
            "total_evaluated": len(stage2_results),
            "aggregate": bake_agg,
            "samples": stage2_results,
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
        raise RuntimeError(f"{len(all_failures)} failures encountered during 3D evaluation")

    return payload


@hydra.main(
    version_base="1.3", config_path="../configs", config_name="rerendering_eval_pbr_3d"
)
def main(config: DictConfig) -> None:
    evaluate(config)


if __name__ == "__main__":
    main()
