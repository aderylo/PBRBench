"""Evaluate screen-space PBR predictions against reference maps."""

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

from src.data.pbr_estimation_dataset_2d import PBREstimationSample2D
from src.methods_2d import Prediction2D
from src.utils import get_pylogger
from src.utils.eval import align_resolutions, load_image, load_mask, write_yaml
from src.utils.metrics import mean_metrics, psnr, rmse

log = get_pylogger(__name__)


@dataclass(frozen=True)
class ImageMetrics:
    """Numerical error metrics between prediction and reference maps."""

    rmse: float
    psnr: float


@dataclass(frozen=True)
class SampleMetrics2D:
    """Direct channel evaluation metrics for one registered sample."""

    albedo: ImageMetrics
    roughness: ImageMetrics
    metallic: ImageMetrics


def evaluate_sample(
    sample: PBREstimationSample2D, pred: Prediction2D
) -> SampleMetrics2D:
    """Explicitly evaluate Albedo, Roughness, and Metallic maps for one sample."""
    mask = load_mask(sample.mask)

    # 1. Albedo (linear RGB)
    gt_albedo = load_image(sample.albedo, rgb=True, to_linear=True)
    pred_albedo = load_image(pred.albedo, rgb=True, to_linear=True)
    pred_albedo, gt_albedo, mask_albedo = align_resolutions(
        pred_albedo, gt_albedo, mask
    )

    # 2. Roughness (linear grayscale)
    gt_roughness = load_image(sample.roughness, rgb=False)
    pred_roughness = load_image(pred.roughness, rgb=False)
    pred_roughness, gt_roughness, mask_roughness = align_resolutions(
        pred_roughness, gt_roughness, mask
    )

    # 3. Metallic (linear grayscale)
    gt_metallic = load_image(sample.metallic, rgb=False)
    pred_metallic = load_image(pred.metallic, rgb=False)
    pred_metallic, gt_metallic, mask_metallic = align_resolutions(
        pred_metallic, gt_metallic, mask
    )

    return SampleMetrics2D(
        albedo=ImageMetrics(
            rmse=rmse(pred_albedo, gt_albedo, mask_albedo),
            psnr=psnr(pred_albedo, gt_albedo, mask_albedo),
        ),
        roughness=ImageMetrics(
            rmse=rmse(pred_roughness, gt_roughness, mask_roughness),
            psnr=psnr(pred_roughness, gt_roughness, mask_roughness),
        ),
        metallic=ImageMetrics(
            rmse=rmse(pred_metallic, gt_metallic, mask_metallic),
            psnr=psnr(pred_metallic, gt_metallic, mask_metallic),
        ),
    )



def evaluate(config: DictConfig) -> dict:
    """Evaluate 2D predictions against ground truth dataset."""
    log.info(f"Resolving predictions directory: {config.predictions_dir}")
    predictions_dir = Path(config.predictions_dir).resolve()

    log.info(f"Instantiating dataset <{config.data._target_}>")
    dataset = instantiate(config.data)
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

    results: dict[str, dict] = {}
    failures: dict[str, str] = {}

    for sample in tqdm(dataset, desc="Direct PBR eval", unit="sample"):
        sid = sample.sample_id
        pred = discovered_predictions.get(sid)

        if not pred:
            failures[sid] = f"Prediction directory missing: {predictions_dir / sid}"
            continue

        try:
            metrics = evaluate_sample(sample, pred)
            results[sid] = {
                "source": sample.source or "",
                "object_id": sample.object_id,
                "view_id": sample.view_id,
                "light_id": sample.light_id,
                "metrics": asdict(metrics),
            }
        except (FileNotFoundError, ValueError, OSError, TypeError) as error:
            failures[sid] = str(error)


    # Aggregate overall mean metrics
    metrics_list = [r["metrics"] for r in results.values()]
    aggregate = mean_metrics(metrics_list)

    payload = {
        "evaluation": "pbr_2d_direct",
        "predictions_dir": str(predictions_dir),
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
        else predictions_dir.parent / "metrics_direct.yaml"
    )
    log.info(f"Writing evaluation metrics to {output_file}")
    write_yaml(output_file, payload)
    log.info(f"Aggregate metrics: {aggregate}")

    if failures and bool(config.get("strict", False)):
        raise RuntimeError(
            f"{len(failures)} failures out of {len(dataset)} samples"
        )

    return payload


@hydra.main(
    version_base="1.3", config_path="../configs", config_name="eval_pbr_2d_direct"
)
def main(config: DictConfig) -> None:
    evaluate(config)


if __name__ == "__main__":
    main()

