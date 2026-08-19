"""Helpers used in evaluation scripts."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, is_dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import yaml
from PIL import Image

from src.data.envmaps import EnvMapSpec
from src.data.pbr_estimation_dataset_2d import (
    PBREstimationSample2D,
    ViewMetadata,
)
from src.data.pbr_estimation_dataset_3d import PBREstimationSample3D
from src.methods_2d import Prediction2D
from src.methods_3d import Prediction3D
from src.utils.metrics import (
    LPIPSMetric,
    mae,
    psnr,
    rmse,
    ssim,
)
from src.utils.rerender_2d_orchestrator import RenderItem2D
from src.utils.rerender_3d_orchestrator import RenderItem3D

CHANNELS = ("albedo", "roughness", "metallic")


# -----------------------------------------------------------------------------
# 2D Evaluation Utils
# -----------------------------------------------------------------------------


def scan_pbr_predictions_dir_2d(
    predictions_dir: Path | str,
) -> dict[str, Prediction2D]:
    """Scan and discover 2D predictions in a directory keyed by sample id."""
    path = Path(predictions_dir)
    if not path.is_dir():
        return {}
    return {
        d.name: Prediction2D.from_dir(d)
        for d in sorted(path.iterdir())
        if d.is_dir()
    }


def gt_sample2d_to_render_item(
    sample: PBREstimationSample2D,
    envmap: EnvMapSpec,
    rerenders_dir: Path,
) -> RenderItem2D:
    """Construct RenderItem2D for one ground-truth relighting render."""
    meta = sample.view_metadata
    return RenderItem2D(
        item_id=f"{sample.sample_id}__{envmap.id}",
        asset_path=Path(meta.asset_path),
        normalization=meta.normalization_source_to_world,
        camera=meta.camera,
        albedo=sample.albedo,
        roughness=sample.roughness,
        metallic=sample.metallic,
        envmap=envmap,
        output_path=rerenders_dir / "gt" / sample.sample_id / f"{envmap.id}.png",
    )


def pred2d_to_render_item(
    pred: Prediction2D,
    view_metadata: ViewMetadata,
    envmap: EnvMapSpec,
    rerenders_dir: Path,
) -> RenderItem2D:
    """Construct RenderItem2D for one prediction relighting render."""
    return RenderItem2D(
        item_id=f"{pred.sample_id}__{envmap.id}",
        asset_path=Path(view_metadata.asset_path),
        normalization=view_metadata.normalization_source_to_world,
        camera=view_metadata.camera,
        albedo=Path(pred.albedo),
        roughness=Path(pred.roughness),
        metallic=Path(pred.metallic),
        envmap=envmap,
        output_path=rerenders_dir / "pred" / pred.sample_id / f"{envmap.id}.png",
    )


# -----------------------------------------------------------------------------
# 3D Evaluation Utils
# -----------------------------------------------------------------------------


def scan_pbr_predictions_dir_3d(
    predictions_dir: Path | str,
) -> dict[str, Prediction3D]:
    """Scan and discover 3D predictions in a directory keyed by sample id."""
    path = Path(predictions_dir)
    if not path.is_dir():
        return {}
    return {
        d.name: Prediction3D.from_dir(d)
        for d in sorted(path.iterdir())
        if d.is_dir()
    }


def gt_sample3d_to_render_item(
    sample: PBREstimationSample3D,
    envmap: EnvMapSpec,
    rerenders_dir: Path,
    *,
    mode: Literal["render", "bake"] = "render",
) -> RenderItem3D:
    """Construct RenderItem3D for one 3D ground-truth relighting render or bake task."""
    gt_asset = sample.mesh_path
    if not gt_asset.is_file() and sample.asset_path is not None:
        gt_asset = sample.asset_path

    view_tag = (
        sample.reference_view.parent.parent.name
        if sample.reference_view is not None
        else "default"
    )
    if mode == "render":
        output_path = rerenders_dir / "render" / "gt" / sample.object_id / view_tag / f"{envmap.id}.png"
        camera = (
            sample.view_metadata.camera
            if sample.view_metadata is not None
            else None
        )
    else:
        output_path = rerenders_dir / "bake" / "gt" / sample.object_id / f"{envmap.id}.png"
        camera = None

    normalization = (
        sample.view_metadata.normalization_source_to_world
        if sample.view_metadata is not None
        else None
    )

    return RenderItem3D(
        item_id=f"gt__{sample.sample_id}__{envmap.id}__{mode}",
        mesh_path=gt_asset,
        normalization=normalization,
        camera=camera,
        envmap=envmap,
        output_path=output_path,
        mode=mode,
    )


def pred3d_to_render_item(
    pred: Prediction3D,
    sample: PBREstimationSample3D,
    envmap: EnvMapSpec,
    rerenders_dir: Path,
    *,
    mode: Literal["render", "bake"] = "render",
) -> RenderItem3D:
    """Construct RenderItem3D for one 3D prediction relighting render or bake task."""
    if pred.pbr_asset_glb is None:
        raise ValueError(f"Prediction for {pred.sample_id} does not have a pbr_asset_glb")

    camera = (
        sample.view_metadata.camera
        if (mode == "render" and sample.view_metadata is not None)
        else None
    )
    normalization = (
        sample.view_metadata.normalization_source_to_world
        if sample.view_metadata is not None
        else None
    )
    output_path = rerenders_dir / mode / "pred" / sample.sample_id / f"{envmap.id}.png"

    return RenderItem3D(
        item_id=f"pred__{sample.sample_id}__{envmap.id}__{mode}",
        mesh_path=Path(pred.pbr_asset_glb),
        normalization=normalization,
        camera=camera,
        envmap=envmap,
        output_path=output_path,
        mode=mode,
    )


# -----------------------------------------------------------------------------
# Common Evaluation Utils
# -----------------------------------------------------------------------------


def srgb_to_linear(value: np.ndarray) -> np.ndarray:
    """Convert sRGB values in [0, 1] to linear RGB."""
    return np.where(
        value <= 0.04045,
        value / 12.92,
        ((value + 0.055) / 1.055) ** 2.4,
    ).astype(np.float32)


def load_image(
    path: Path | str,
    *,
    rgb: bool = False,
    to_linear: bool = False,
) -> np.ndarray:
    """Load an image as float32 in [0, 1], with optional sRGB to linear conversion."""
    with Image.open(path) as image:
        array = (
            np.asarray(image.convert("RGB" if rgb else "L"), dtype=np.float32)
            / 255.0
        )
    if to_linear and rgb:
        array = srgb_to_linear(array)
    return array


def load_mask(path: Path | str) -> np.ndarray:
    """Load a binary foreground mask from a grayscale image."""
    with Image.open(path) as image:
        return np.asarray(image.convert("L")) > 0


def load_alpha(path: Path | str) -> np.ndarray:
    """Load a binary foreground mask from an image alpha channel."""
    with Image.open(path) as image:
        return np.asarray(image.convert("RGBA").getchannel("A")) > 127


def resize_image(
    image: np.ndarray,
    target_shape: tuple[int, int],
    *,
    resample: Image.Resampling = Image.Resampling.BILINEAR,
) -> np.ndarray:
    """Resize a 2D or 3D float image array to target_shape (H, W)."""
    if image.shape[:2] == target_shape:
        return image

    ht, wt = target_shape
    orig_dtype = image.dtype
    if image.ndim == 2:
        resized = Image.fromarray(image.astype(np.float32), mode="F").resize(
            (wt, ht), resample=resample
        )
        return np.asarray(
            resized,
            dtype=orig_dtype if np.issubdtype(orig_dtype, np.floating) else np.float32,
        )
    if image.ndim == 3:
        channels = [
            np.asarray(
                Image.fromarray(image[..., c].astype(np.float32), mode="F").resize(
                    (wt, ht), resample=resample
                ),
                dtype=orig_dtype if np.issubdtype(orig_dtype, np.floating) else np.float32,
            )
            for c in range(image.shape[2])
        ]
        return np.stack(channels, axis=-1)
    raise ValueError(f"unsupported image ndim {image.ndim}")


def resize_mask(
    mask: np.ndarray,
    target_shape: tuple[int, int],
) -> np.ndarray:
    """Resize a boolean mask to target_shape (H, W) using nearest neighbor."""
    if mask.shape[:2] == target_shape:
        return mask
    ht, wt = target_shape
    resized = Image.fromarray(mask.astype(bool)).resize(
        (wt, ht), resample=Image.Resampling.NEAREST
    )
    return np.asarray(resized, dtype=bool)


def align_resolutions(
    pred: np.ndarray,
    target: np.ndarray,
    mask: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """If prediction and target resolutions differ but aspect ratios match, downsample the higher resolution."""
    if pred.shape[:2] == target.shape[:2]:
        return pred, target, mask

    hp, wp = pred.shape[:2]
    ht, wt = target.shape[:2]

    # Validate aspect ratios
    ar_pred = hp / wp
    ar_target = ht / wt
    if not math.isclose(ar_pred, ar_target, rel_tol=1e-2, abs_tol=1e-2):
        raise ValueError(
            f"Aspect ratio mismatch cannot be aligned: prediction {pred.shape[:2]} vs target {target.shape[:2]}"
        )

    if hp * wp > ht * wt:
        pred = resize_image(pred, (ht, wt))
        if mask is not None and mask.shape[:2] != (ht, wt):
            mask = resize_mask(mask, (ht, wt))
    else:
        target = resize_image(target, (hp, wp))
        if mask is not None and mask.shape[:2] != (hp, wp):
            mask = resize_mask(mask, (hp, wp))

    return pred, target, mask


def write_yaml(path: Path | str, payload: Any) -> None:
    """Serialize a mapping or dataclass payload as readable YAML."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    serializable_payload = (
        asdict(payload) if is_dataclass(payload) else payload
    )
    p.write_text(yaml.safe_dump(serializable_payload, sort_keys=False))


# -----------------------------------------------------------------------------
# Rerendering Evaluators
# -----------------------------------------------------------------------------


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


class RerenderingImageEvaluator:
    """Evaluates image-space appearance metrics between prediction and reference renders."""

    def __init__(
        self,
        device: str = "cpu",
        backbone: str = "alex",
        model_cache_dir: Path | str = "data/checkpoints/torch",
    ) -> None:
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


class RerenderingUVEvaluator:
    """Evaluates texture-space metrics between baked prediction and reference UV maps."""

    def evaluate(
        self,
        pred_path: Path,
        gt_path: Path,
        uv_mask_path: Path | None = None,
    ) -> dict[str, float]:
        """Evaluate texture-space metrics on one baked UV texture pair."""
        prediction = load_image(pred_path, rgb=True)
        target = load_image(gt_path, rgb=True)
        mask = (
            load_mask(uv_mask_path)
            if uv_mask_path is not None and uv_mask_path.is_file()
            else None
        )

        prediction, target, mask = align_resolutions(prediction, target, mask)

        return asdict(
            BakeMetrics(
                rmse=rmse(prediction, target, mask),
                psnr=psnr(prediction, target, mask),
                mae=mae(prediction, target, mask),
                ssim=ssim(prediction, target, mask),
            )
        )

