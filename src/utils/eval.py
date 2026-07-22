"""Shared evaluation I/O, report serialization, and prediction discovery."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import asdict, dataclass, is_dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from PIL import Image

CHANNELS = ("albedo", "roughness", "metallic")


@dataclass(frozen=True)
class Prediction:
    """The persisted channel files for one prediction sample."""

    sample_id: str
    directory: Path
    channels: dict[str, Path]

    def missing_channels(self) -> tuple[str, ...]:
        """Return required channels that are absent from this prediction."""
        return tuple(name for name, path in self.channels.items() if not path.is_file())


def scan_predictions(
    predictions_dir: Path, required_channels: Iterable[str]
) -> dict[str, Prediction]:
    """Scan canonical ``<predictions_dir>/<sample_id>/<channel>.png`` outputs."""
    if not predictions_dir.is_dir():
        raise FileNotFoundError(
            f"Prediction directory does not exist: {predictions_dir}"
        )

    channel_names = tuple(required_channels)
    return {
        directory.name: Prediction(
            sample_id=directory.name,
            directory=directory,
            channels={
                channel: directory / f"{channel}.png" for channel in channel_names
            },
        )
        for directory in sorted(predictions_dir.iterdir())
        if directory.is_dir()
    }


def load_image(path: Path, *, rgb: bool) -> np.ndarray:
    """Load an image as float32 in [0, 1]."""
    with Image.open(path) as image:
        array = np.asarray(image.convert("RGB" if rgb else "L"), dtype=np.float32)
    return array / 255.0


def load_mask(path: Path) -> np.ndarray:
    """Load a binary foreground mask from a grayscale image."""
    with Image.open(path) as image:
        return np.asarray(image.convert("L")) > 0


def load_alpha(path: Path) -> np.ndarray:
    """Load a binary foreground mask from an image alpha channel."""
    with Image.open(path) as image:
        return np.asarray(image.convert("RGBA").getchannel("A")) > 127


def srgb_to_linear(value: np.ndarray) -> np.ndarray:
    """Convert sRGB values in [0, 1] to linear RGB."""
    return np.where(
        value <= 0.04045,
        value / 12.92,
        ((value + 0.055) / 1.055) ** 2.4,
    ).astype(np.float32)


def write_yaml(path: Path, payload: Any) -> None:
    """Serialize a mapping or dataclass payload as readable YAML."""
    path.parent.mkdir(parents=True, exist_ok=True)
    serializable_payload = asdict(payload) if is_dataclass(payload) else payload
    path.write_text(yaml.safe_dump(serializable_payload, sort_keys=False))
