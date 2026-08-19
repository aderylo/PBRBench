"""Relight screen-space PBR predictions in Blender and compare the renders."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import hydra
import rootutils
from hydra.utils import instantiate
from omegaconf import DictConfig
from tqdm.auto import tqdm

PROJECT_ROOT = rootutils.setup_root(
    __file__, indicator=".project_root", pythonpath=True
)

from src.data.envmaps import EnvMapDataset
from src.data.pbr_estimation_dataset_2d import (
    PBREstimationDataset2D,
)
from src.utils import get_pylogger
from src.utils.eval import (
    align_resolutions,
    gt_sample2d_to_render_item,
    load_alpha,
    load_image,
    pred2d_to_render_item,
    scan_pbr_predictions_dir_2d,
    write_yaml,
)
from src.utils.metrics import (
    LPIPSMetric,
    mean_metrics,
    psnr,
    rmse,
    ssim,
)
from src.utils.rerender_2d_orchestrator import (
    Rerenderer2D,
)

log = get_pylogger(__name__)


@dataclass(frozen=True)
class RenderMetrics:
    """Image-space comparison metrics between prediction and reference renders."""

    rmse: float
    psnr: float
    ssim: float
    lpips: float


class RerenderingEvaluator:
    """Evaluates image-space appearance metrics between prediction and reference renders."""

    def __init__(self, device: str, backbone: str, model_cache_dir: Path | str) -> None:
        self.lpips = LPIPSMetric(
            str(device), str(backbone), Path(model_cache_dir).resolve()
        )

    def evaluate(self, pred_path: Path, gt_path: Path) -> dict[str, float]:
        """Evaluate image metrics on one relit viewpoint/target pair."""
        prediction = load_image(pred_path, rgb=True)
        target = load_image(gt_path, rgb=True)
        mask = load_alpha(gt_path)

        prediction, target, mask = align_resolutions(prediction, target, mask)

        return asdict(
            RenderMetrics(
                rmse=rmse(prediction, target, mask),
                psnr=psnr(prediction, target, mask),
                ssim=ssim(prediction, target, mask),
                lpips=self.lpips(prediction, target, mask),
            )
        )


def evaluate(config: DictConfig) -> dict:
    """Relight registered predictions in Blender and evaluate rendered appearance."""
    log.info(f"Resolving predictions directory: {config.predictions_dir}")
    predictions_dir = Path(config.predictions_dir).resolve()

    log.info(f"Instantiating dataset <{config.data._target_}>")
    dataset: PBREstimationDataset2D = instantiate(config.data)
    log.info(f"Dataset loaded with {len(dataset)} samples")

    log.info(f"Scanning predictions in {predictions_dir}")
    predictions = scan_pbr_predictions_dir_2d(predictions_dir)
    log.info(f"Discovered {len(predictions)} prediction directories")

    log.info(f"Instantiating lighting dataset <{config.lighting._target_}>")
    envmaps: EnvMapDataset = instantiate(config.lighting)

    matched_samples = [
        (sample, predictions[sample.sample_id])
        for sample in dataset
        if sample.sample_id in predictions
    ]
    log.info(f"Matched {len(matched_samples)}/{len(dataset)} predictions with GT")

    rerenders_dir = predictions_dir.parent / "rerenders"
    blender_log_path = predictions_dir.parent / "blender_relight.log"

    gt_render_items = [
        gt_sample2d_to_render_item(sample, envmap, rerenders_dir)
        for sample, _ in matched_samples
        for envmap in envmaps
    ]
    pred_render_items = [
        pred2d_to_render_item(pred, sample.view_metadata, envmap, rerenders_dir)
        for sample, pred in matched_samples
        for envmap in envmaps
    ]

    log.info(f"Instantiating rerenderer <{config.rerenderer._target_}>")
    rerenderer: Rerenderer2D = instantiate(config.rerenderer)
    rerenderer.render(
        items=gt_render_items + pred_render_items,
        working_dir=rerenders_dir,
        blender_log_path=blender_log_path,
    )

    log.info(f"Instantiating evaluator <{config.evaluator._target_}>")
    evaluator: RerenderingEvaluator = instantiate(config.evaluator)

    results: dict[str, dict] = {}
    failures: dict[str, str] = {}
    all_metrics: list[dict[str, float]] = []

    for gt_rerender, pred_rerender in tqdm(
        zip(gt_render_items, pred_render_items),
        total=len(gt_render_items),
        desc="Indirect PBR eval",
        unit="render",
    ):
        try:
            m = evaluator.evaluate(pred_rerender.output_path, gt_rerender.output_path)
            all_metrics.append(m)
            results[pred_rerender.item_id] = m
        except (FileNotFoundError, ValueError, OSError, TypeError) as error:
            failures[pred_rerender.item_id] = str(error)

    aggregate = mean_metrics(all_metrics)

    payload = {
        "evaluation": "pbr_2d_indirect",
        "predictions_dir": str(predictions_dir),
        "target_envmaps": [e.id for e in envmaps],
        "total_requested": len(gt_render_items),
        "total_evaluated": len(results),
        "total_failed": len(failures),
        "aggregate": aggregate,
        "results": results,
        "failures": failures,
    }

    output_file = (
        Path(config.output_file).resolve()
        if config.get("output_file")
        else predictions_dir.parent / "metrics_indirect.yaml"
    )
    log.info(f"Writing evaluation metrics to {output_file}")
    write_yaml(output_file, payload)
    log.info(f"Aggregate metrics: {aggregate}")

    if failures and bool(config.get("strict", False)):
        raise RuntimeError(f"{len(failures)} failures out of {len(gt_render_items)} renders")

    return payload


@hydra.main(
    version_base="1.3", config_path="../configs", config_name="rerendering_eval_pbr_2d"
)
def main(config: DictConfig) -> None:
    evaluate(config)


if __name__ == "__main__":
    main()
