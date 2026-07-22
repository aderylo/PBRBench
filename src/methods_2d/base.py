"""Base contract and standard utilities for screen-space material estimators."""

from __future__ import annotations

import shutil
from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from PIL import Image

from src.data.pbr_estimation_dataset_2d import PBREstimationSample2D


@dataclass(frozen=True)
class Prediction2D:
    """Predicted 2D material maps for one sample."""

    albedo: Path
    roughness: Path
    metallic: Path
    normal: Path | None = None
    artifacts: Mapping[str, Path] = field(default_factory=dict)


ImageInput = Any  # Image.Image | np.ndarray | torch.Tensor | Path


def save_image_artifact(image_input: ImageInput, target_path: Path) -> Path:
    """Save PIL Image, numpy array, torch Tensor, or existing file Path to target_path."""
    if isinstance(image_input, (str, Path)):
        src_path = Path(image_input)
        if src_path.resolve() != target_path.resolve():
            target_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src_path, target_path)
        return target_path

    target_path.parent.mkdir(parents=True, exist_ok=True)

    if hasattr(image_input, "detach"):  # torch.Tensor
        image_input = image_input.detach().cpu().numpy()

    # If image is numpy array
    if hasattr(image_input, "ndim") and hasattr(image_input, "shape"):
        import numpy as np

        if (
            image_input.ndim == 3
            and image_input.shape[0] in (1, 3, 4)
            and image_input.shape[0] < image_input.shape[1]
        ):
            # Convert CHW -> HWC
            image_input = image_input.transpose(1, 2, 0)
        if image_input.dtype != np.uint8:
            if image_input.max() <= 1.0:
                image_input = (image_input * 255.0).clip(0, 255).astype(np.uint8)
            else:
                image_input = image_input.clip(0, 255).astype(np.uint8)
        if image_input.ndim == 3 and image_input.shape[2] == 1:
            image_input = image_input.squeeze(2)
        image_input = Image.fromarray(image_input)

    if isinstance(image_input, Image.Image):
        image_input.save(target_path)
        return target_path

    raise TypeError(f"Unsupported image type for saving: {type(image_input)}")


class BaseMaterialEstimator2D(ABC):
    """Common interface implemented by every 2D material estimator."""

    def __init__(
        self,
        *,
        name: str,
        project_root: str | Path,
        repo_root: str | Path,
    ) -> None:
        self.name = name
        self.project_root = Path(project_root).resolve()
        self.repo_root = self.resolve_path(repo_root)

    def resolve_path(self, path: str | Path) -> Path:
        path = Path(path)
        return (
            path.resolve()
            if path.is_absolute()
            else (self.project_root / path).resolve()
        )

    def setup(self) -> None:
        """Load the upstream implementation and model weights."""
        if not self.repo_root.is_dir():
            raise FileNotFoundError(
                f"{self.name} repository not found: {self.repo_root}"
            )

    def teardown(self) -> None:
        """Release estimator-owned resources, if any."""

    @staticmethod
    def require_file(path: Path, description: str) -> Path:
        if not path.is_file():
            raise FileNotFoundError(f"Missing {description}: {path}")
        return path

    @staticmethod
    def save_prediction(
        sample_dir: Path,
        albedo: ImageInput,
        roughness: ImageInput,
        metallic: ImageInput,
        normal: ImageInput | None = None,
        artifacts: Mapping[str, ImageInput] | None = None,
    ) -> Prediction2D:
        """Standardized helper to save predicted material channels for a sample and return Prediction2D."""
        sample_dir.mkdir(parents=True, exist_ok=True)

        saved_albedo = save_image_artifact(albedo, sample_dir / "albedo.png")
        saved_roughness = save_image_artifact(roughness, sample_dir / "roughness.png")
        saved_metallic = save_image_artifact(metallic, sample_dir / "metallic.png")

        saved_normal = None
        if normal is not None:
            saved_normal = save_image_artifact(normal, sample_dir / "normal.png")

        saved_artifacts: dict[str, Path] = {
            "albedo": saved_albedo,
            "roughness": saved_roughness,
            "metallic": saved_metallic,
        }
        if saved_normal is not None:
            saved_artifacts["normal"] = saved_normal

        if artifacts:
            for name, art_input in artifacts.items():
                if name in ("albedo", "roughness", "metallic", "normal"):
                    continue
                ext = (
                    ".png"
                    if not isinstance(art_input, (str, Path))
                    or not Path(art_input).suffix
                    else Path(art_input).suffix
                )
                saved_artifacts[name] = save_image_artifact(
                    art_input, sample_dir / f"{name}{ext}"
                )

        return Prediction2D(
            albedo=saved_albedo,
            roughness=saved_roughness,
            metallic=saved_metallic,
            normal=saved_normal,
            artifacts=saved_artifacts,
        )

    @abstractmethod
    def predict(
        self,
        samples: Sequence[PBREstimationSample2D],
        output_dir: Path,
    ) -> Mapping[str, Prediction2D]:
        """Predict material maps for a non-empty collection of samples."""
