"""Evaluation I/O, image loading, and serialization helpers."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import asdict, dataclass, is_dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from PIL import Image

CHANNELS = ("albedo", "roughness", "metallic")





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


def write_yaml(path: Path | str, payload: Any) -> None:
    """Serialize a mapping or dataclass payload as readable YAML."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    serializable_payload = (
        asdict(payload) if is_dataclass(payload) else payload
    )
    p.write_text(yaml.safe_dump(serializable_payload, sort_keys=False))

