"""Numerical image metrics used by PBR evaluators."""

from __future__ import annotations

import math
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np


def validate_matching_shapes(
    prediction: np.ndarray,
    target: np.ndarray,
    mask: np.ndarray | None = None,
) -> None:
    """Validate that prediction, target, and optional mask shapes match."""
    if prediction.shape != target.shape:
        raise ValueError(
            f"shape mismatch: prediction {prediction.shape}, target {target.shape}"
        )
    if mask is not None:
        if mask.shape != target.shape[:2]:
            raise ValueError(
                f"mask shape {mask.shape} does not match image {target.shape}"
            )
        if not mask.any():
            raise ValueError("foreground mask is empty")


def compute_difference(
    prediction: np.ndarray,
    target: np.ndarray,
    mask: np.ndarray | None = None,
) -> np.ndarray:
    """Return pixel difference array (masked if mask is provided)."""
    validate_matching_shapes(prediction, target, mask)
    return (
        prediction[mask] - target[mask]
        if mask is not None
        else prediction - target
    )


def rmse(
    prediction: np.ndarray,
    target: np.ndarray,
    mask: np.ndarray | None = None,
) -> float:
    """Compute Root Mean Squared Error (RMSE), optionally masked."""
    diff = compute_difference(prediction, target, mask=mask)
    return float(np.sqrt(np.mean(diff * diff)))


def psnr(
    prediction: np.ndarray,
    target: np.ndarray,
    mask: np.ndarray | None = None,
) -> float:
    """Compute Peak Signal-to-Noise Ratio (PSNR) in dB, optionally masked."""
    val = rmse(prediction, target, mask=mask)
    return math.inf if val == 0.0 else float(-20.0 * math.log10(val))


def mae(
    prediction: np.ndarray,
    target: np.ndarray,
    mask: np.ndarray | None = None,
) -> float:
    """Compute Mean Absolute Error (MAE), optionally masked."""
    diff = compute_difference(prediction, target, mask=mask)
    return float(np.mean(np.abs(diff)))


def ssim(
    prediction: np.ndarray,
    target: np.ndarray,
    mask: np.ndarray | None = None,
) -> float:

    """Compute (optionally masked) RGB SSIM with standard Gaussian local statistics."""
    validate_matching_shapes(prediction, target, mask)
    if prediction.ndim != 3:
        raise ValueError("SSIM expects equally shaped RGB images (H, W, C)")

    try:
        from scipy.ndimage import binary_erosion, gaussian_filter
    except ImportError as error:
        raise RuntimeError(
            "Indirect evaluation requires scipy; run `uv sync`."
        ) from error


    sigma = 1.5
    mean_x = gaussian_filter(prediction, sigma=(sigma, sigma, 0))
    mean_y = gaussian_filter(target, sigma=(sigma, sigma, 0))
    variance_x = (
        gaussian_filter(prediction * prediction, sigma=(sigma, sigma, 0))
        - mean_x**2
    )
    variance_y = (
        gaussian_filter(target * target, sigma=(sigma, sigma, 0)) - mean_y**2
    )
    covariance = (
        gaussian_filter(prediction * target, sigma=(sigma, sigma, 0))
        - mean_x * mean_y
    )
    c1 = 0.01**2
    c2 = 0.03**2
    score_map = ((2 * mean_x * mean_y + c1) * (2 * covariance + c2)) / (
        (mean_x**2 + mean_y**2 + c1) * (variance_x + variance_y + c2)
    )

    if mask is not None:
        if not mask.any():
            raise ValueError("foreground mask is empty")
        valid = binary_erosion(mask, iterations=5)
        if not valid.any():
            valid = mask
    else:
        valid = np.ones(prediction.shape[:2], dtype=bool)

    return float(score_map[valid].mean())


class LPIPSMetric:
    """Lazily construct LPIPS because direct evaluation does not need torch."""

    def __init__(
        self,
        device: str = "cpu",
        backbone: str = "alex",
        cache_dir: Path | None = None,
    ) -> None:
        try:
            import lpips
            import torch
        except ImportError as error:
            raise RuntimeError(
                "Indirect evaluation requires torch and lpips; run `uv sync`."
            ) from error
        self.torch = torch
        self.device = device
        if cache_dir is not None:
            cache_dir.mkdir(parents=True, exist_ok=True)
            torch.hub.set_dir(str(cache_dir))
        self.model = lpips.LPIPS(net=backbone).to(device).eval()

    def __call__(
        self,
        prediction: np.ndarray,
        target: np.ndarray,
        mask: np.ndarray | None = None,
    ) -> float:
        if mask is not None:
            rows, columns = np.nonzero(mask)
            if len(rows) == 0:
                raise ValueError("foreground mask is empty")
            y0, y1 = int(rows.min()), int(rows.max()) + 1
            x0, x1 = int(columns.min()), int(columns.max()) + 1
            crop_mask = mask[y0:y1, x0:x1, None]
            pred_crop = prediction[y0:y1, x0:x1] * crop_mask
            target_crop = target[y0:y1, x0:x1] * crop_mask
        else:
            pred_crop = prediction
            target_crop = target

        def tensor(array: np.ndarray):
            value = self.torch.from_numpy(array.copy()).permute(2, 0, 1)[None]
            value = value.to(self.device) * 2.0 - 1.0
            height, width = value.shape[-2:]
            if min(height, width) < 64:
                scale = 64.0 / min(height, width)
                value = self.torch.nn.functional.interpolate(
                    value,
                    size=(round(height * scale), round(width * scale)),
                    mode="bilinear",
                    align_corners=False,
                )
            return value

        with self.torch.inference_mode():
            return float(
                self.model(tensor(pred_crop), tensor(target_crop)).item()
            )



def mean_metrics(items: Iterable[dict[str, Any]]) -> dict[str, Any]:
    """Recursively average matching scalar metrics."""
    items = list(items)
    if not items:
        return {}
    keys = items[0].keys()
    result: dict[str, Any] = {}
    for key in keys:
        values = [item[key] for item in items]
        if isinstance(values[0], dict):
            result[key] = mean_metrics(values)
        else:
            result[key] = float(np.mean(values))
    return result


def metric_statistics(items: Iterable[dict[str, Any]]) -> dict[str, Any]:
    """Recursively summarize scalar metrics across evaluated samples.

    Variance and standard deviation use the population definition (``ddof=0``),
    because the evaluated samples represent the complete selected evaluation
    split rather than a statistical sample of a larger set.
    """
    items = list(items)
    if not items:
        return {}
    keys = items[0].keys()
    result: dict[str, Any] = {}
    for key in keys:
        values = [item[key] for item in items]
        if isinstance(values[0], dict):
            result[key] = metric_statistics(values)
            continue

        array = np.asarray(values, dtype=np.float64)
        result[key] = {
            "count": int(array.size),
            "mean": float(np.mean(array)),
            "variance": float(np.var(array)),
            "stddev": float(np.std(array)),
            "min": float(np.min(array)),
            "median": float(np.median(array)),
            "max": float(np.max(array)),
        }
    return result
