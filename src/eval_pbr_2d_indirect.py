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

from src.data.pbr_estimation_dataset_2d import PBREstimationDataset2D
from src.methods_2d import Prediction2D
from src.utils import get_pylogger
from src.utils.eval import align_resolutions, load_alpha, load_image, write_yaml
from src.utils.metrics import (
    LPIPSMetric,
    mean_metrics,
    psnr,
    rmse,
    ssim,
)
from src.utils.relight_2d import get_relight_working_dir, relight_dataset_2d

log = get_pylogger(__name__)


@dataclass(frozen=True)
class RenderMetrics:
    """Image-space comparison metrics between prediction and reference renders."""

    rmse: float
    psnr: float
    ssim: float
    lpips: float


def evaluate_relit_target(
    pred_path: Path,
    gt_path: Path,
    lpips_metric: LPIPSMetric,
) -> RenderMetrics:
    """Evaluate image metrics on one relit viewpoint/target pair."""
    prediction = load_image(pred_path, rgb=True)
    target = load_image(gt_path, rgb=True)
    mask = load_alpha(gt_path)

    prediction, target, mask = align_resolutions(prediction, target, mask)

    return RenderMetrics(
        rmse=rmse(prediction, target, mask),
        psnr=psnr(prediction, target, mask),
        ssim=ssim(prediction, target, mask),
        lpips=lpips_metric(prediction, target, mask),
    )


def evaluate(config: DictConfig) -> dict:
    """Relight registered predictions in Blender and evaluate rendered appearance."""
    log.info(f"Resolving predictions directory: {config.predictions_dir}")
    predictions_dir = Path(config.predictions_dir).resolve()

    log.info(f"Instantiating dataset <{config.data._target_}>")
    dataset: PBREstimationDataset2D = instantiate(config.data)
    log.info(f"Dataset loaded with {len(dataset)} samples")

    log.info(f"Scanning predictions in {predictions_dir}")
    discovered_predictions = (
        {
            d.name: Prediction2D.from_dir(d)
            for d in sorted(predictions_dir.iterdir())
            if d.is_dir()
        }
        if predictions_dir.is_dir()
        else {}
    )
    log.info(f"Discovered {len(discovered_predictions)} prediction directories")

    save_rerenders = bool(config.get("save_rerenders"))
    blender_log_path = predictions_dir.parent / "blender_relight.log"

    with get_relight_working_dir(predictions_dir, save_rerenders=save_rerenders) as working_dir:
        job_spec, score_paths, failures = relight_dataset_2d(
            config=config,
            dataset=dataset,
            predictions=discovered_predictions,
            working_dir=working_dir,
            blender_log_path=blender_log_path,
        )

        lpips_metric = LPIPSMetric(
            str(config.device),
            str(config.lpips_backbone),
            Path(config.model_cache_dir).resolve(),
        )

        results: dict[str, dict] = {}
        all_target_metrics: list[dict[str, float]] = []
        relight_target_metrics: list[dict[str, float]] = []
        cycle_target_metrics: list[dict[str, float]] = []

        for sample in tqdm(dataset, desc="Indirect PBR eval", unit="sample"):
            sid = sample.sample_id
            target_paths = score_paths.get(sid)
            if not target_paths:
                continue

            try:
                target_results: dict[str, dict[str, float]] = {}
                for target_id, (pred_render, gt_render) in target_paths.items():
                    m = evaluate_relit_target(pred_render, gt_render, lpips_metric)
                    m_dict = asdict(m)
                    target_results[target_id] = m_dict

                    all_target_metrics.append(m_dict)
                    if target_id == sample.light_id:
                        cycle_target_metrics.append(m_dict)
                    else:
                        relight_target_metrics.append(m_dict)

                results[sid] = {
                    "source": sample.source or "",
                    "object_id": sample.object_id,
                    "view_id": sample.view_id,
                    "light_id": sample.light_id,
                    "metrics": mean_metrics(list(target_results.values())),
                    "targets": target_results,
                }
            except (FileNotFoundError, ValueError, OSError, TypeError) as error:
                failures[sid] = str(error)

    if not cycle_target_metrics:
        log.warning("No cycle-consistency targets evaluated (source light not in target envmaps).")

    aggregate = {
        "relight": mean_metrics(relight_target_metrics) if relight_target_metrics else mean_metrics(all_target_metrics),
        "cycle": mean_metrics(cycle_target_metrics) if cycle_target_metrics else None,
        "overall": mean_metrics(all_target_metrics),
    }

    payload = {
        "evaluation": "pbr_2d_indirect",
        "predictions_dir": str(predictions_dir),
        "target_envmaps": [target["id"] for target in job_spec["targets"]],
        "total_requested": len(dataset),
        "total_evaluated": len(results),
        "total_failed": len(failures),
        "aggregate": aggregate,
        "samples": results,
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
        raise RuntimeError(f"{len(failures)} failures out of {len(dataset)} samples")

    return payload


@hydra.main(
    version_base="1.3", config_path="../configs", config_name="eval_pbr_2d_indirect"
)
def main(config: DictConfig) -> None:
    evaluate(config)


if __name__ == "__main__":
    main()
