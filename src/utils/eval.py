"""Helpers used in evaluation scripts."""

from __future__ import annotations

import math
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from PIL import Image

from src.data.envmaps import EnvMapSpec
from src.data.pbr_estimation_dataset_2d import (
    PBREstimationSample2D,
    ViewMetadata,
)
from src.methods_2d import Prediction2D
from src.methods_3d import Prediction3D
from src.utils.rerender_2d_orchestrator import RenderItem2D

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
