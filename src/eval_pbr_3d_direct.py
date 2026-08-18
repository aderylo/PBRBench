"""Evaluate 3D PBR predictions against reference ground-truth PBR maps."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import hydra
import numpy as np
import rootutils
from hydra.utils import instantiate
from omegaconf import DictConfig
from PIL import Image
from tqdm.auto import tqdm

PROJECT_ROOT = rootutils.setup_root(
    __file__, indicator=".project_root", pythonpath=True
)

from src.data.pbr_estimation_dataset_3d import PBREstimationSample3D
from src.methods_3d import Prediction3D
from src.utils import get_pylogger
from src.utils.eval import load_image, load_mask, srgb_to_linear, write_yaml
from src.utils.glb import extract_pbr_textures
from src.utils.metrics import mae, mean_metrics, psnr, rmse

log = get_pylogger(__name__)


@dataclass(frozen=True)
class ImageMetrics:
    """Numerical error metrics between prediction and reference maps."""

    rmse: float
    psnr: float
    mae: float


@dataclass(frozen=True)
class SampleMetrics3D:
    """Direct channel evaluation metrics for one registered 3D sample."""

    albedo: ImageMetrics
    roughness: ImageMetrics
    metallic: ImageMetrics


def load_prediction_textures(pred: Prediction3D) -> dict[str, np.ndarray]:
    """Load or extract predicted PBR textures (albedo in linear RGB, roughness/metallic in [0,1])."""
    # 1. If explicit channel files exist on disk
    if (
        isinstance(pred.albedo, Path)
        and pred.albedo.is_file()
        and isinstance(pred.roughness, Path)
        and pred.roughness.is_file()
        and isinstance(pred.metallic, Path)
        and pred.metallic.is_file()
    ):
        return {
            "albedo": load_image(pred.albedo, rgb=True, to_linear=True),
            "roughness": load_image(pred.roughness, rgb=False),
            "metallic": load_image(pred.metallic, rgb=False),
        }

    # 2. If GLB asset is present, extract textures directly
    if pred.pbr_asset_glb is not None and Path(pred.pbr_asset_glb).is_file():
        extracted = extract_pbr_textures(pred.pbr_asset_glb)
        albedo_arr = (
            np.asarray(extracted["albedo"].convert("RGB"), dtype=np.float32) / 255.0
        )
        return {
            "albedo": srgb_to_linear(albedo_arr),
            "roughness": (
                np.asarray(extracted["roughness"].convert("L"), dtype=np.float32)
                / 255.0
            ),
            "metallic": (
                np.asarray(extracted["metallic"].convert("L"), dtype=np.float32)
                / 255.0
            ),
        }

    raise FileNotFoundError(
        f"Prediction {pred.sample_id} has neither channel image files nor a valid GLB asset"
    )


def _resize_if_needed(
    image: np.ndarray, target_shape: tuple[int, int], *, is_mask: bool = False
) -> np.ndarray:
    """Resize image or mask to target_shape (H, W) if necessary."""
    if image.shape[:2] == target_shape:
        return image

    if is_mask:
        pil_img = Image.fromarray(image.astype(bool))
        resized = pil_img.resize(
            (target_shape[1], target_shape[0]), resample=Image.Resampling.NEAREST
        )
        return np.asarray(resized) > 0

    pil_img = Image.fromarray((np.clip(image, 0.0, 1.0) * 255.0).astype(np.uint8))
    resized = pil_img.resize(
        (target_shape[1], target_shape[0]), resample=Image.Resampling.BILINEAR
    )
    return np.asarray(resized, dtype=np.float32) / 255.0


def evaluate_sample(
    sample: PBREstimationSample3D, pred: Prediction3D
) -> SampleMetrics3D:
    """Explicitly evaluate Albedo, Roughness, and Metallic maps for one 3D sample."""
    if sample.albedo is None or not sample.albedo.is_file():
        raise FileNotFoundError(
            f"ground-truth albedo is missing for {sample.sample_id}"
        )
    if sample.roughness is None or not sample.roughness.is_file():
        raise FileNotFoundError(
            f"ground-truth roughness is missing for {sample.sample_id}"
        )
    if sample.metallic is None or not sample.metallic.is_file():
        raise FileNotFoundError(
            f"ground-truth metallic is missing for {sample.sample_id}"
        )

    gt_albedo = load_image(sample.albedo, rgb=True, to_linear=True)
    gt_roughness = load_image(sample.roughness, rgb=False)
    gt_metallic = load_image(sample.metallic, rgb=False)

    target_shape = gt_albedo.shape[:2]

    # Load ground truth UV mask if present, otherwise fallback to full mask
    if sample.uv_mask is not None and sample.uv_mask.is_file():
        mask = _resize_if_needed(load_mask(sample.uv_mask), target_shape, is_mask=True)
    else:
        mask = np.ones(target_shape, dtype=bool)

    pred_textures = load_prediction_textures(pred)
    pred_albedo = _resize_if_needed(pred_textures["albedo"], target_shape)
    pred_roughness = _resize_if_needed(pred_textures["roughness"], target_shape)
    pred_metallic = _resize_if_needed(pred_textures["metallic"], target_shape)

    return SampleMetrics3D(
        albedo=ImageMetrics(
            rmse=rmse(pred_albedo, gt_albedo, mask),
            psnr=psnr(pred_albedo, gt_albedo, mask),
            mae=mae(pred_albedo, gt_albedo, mask),
        ),
        roughness=ImageMetrics(
            rmse=rmse(pred_roughness, gt_roughness, mask),
            psnr=psnr(pred_roughness, gt_roughness, mask),
            mae=mae(pred_roughness, gt_roughness, mask),
        ),
        metallic=ImageMetrics(
            rmse=rmse(pred_metallic, gt_metallic, mask),
            psnr=psnr(pred_metallic, gt_metallic, mask),
            mae=mae(pred_metallic, gt_metallic, mask),
        ),
    )


def evaluate(config: DictConfig) -> dict:
    """Evaluate 3D predictions against ground truth dataset."""
    log.info(f"Resolving predictions directory: {config.predictions_dir}")
    predictions_dir = Path(config.predictions_dir).resolve()

    log.info(f"Instantiating dataset <{config.data._target_}>")
    dataset = instantiate(config.data)
    log.info(f"Dataset loaded with {len(dataset)} samples")

    log.info(f"Scanning predictions in {predictions_dir}")
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

    results: dict[str, dict] = {}
    failures: dict[str, str] = {}

    for sample in tqdm(dataset, desc="Direct 3D PBR eval", unit="sample"):
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
                "texture_id": sample.texture_id,
                "metrics": asdict(metrics),
            }
        except (FileNotFoundError, ValueError, OSError, TypeError) as error:
            failures[sid] = str(error)

    # Aggregate overall mean metrics
    metrics_list = [r["metrics"] for r in results.values()]
    aggregate = mean_metrics(metrics_list)

    payload = {
        "evaluation": "pbr_3d_direct",
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
        else predictions_dir.parent / "metrics_direct_3d.yaml"
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
    version_base="1.3", config_path="../configs", config_name="eval_pbr_3d_direct"
)
def main(config: DictConfig) -> None:
    evaluate(config)


if __name__ == "__main__":
    main()
