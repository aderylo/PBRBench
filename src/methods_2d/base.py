"""Base contract and standard utilities for screen-space material estimators."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from PIL import Image

from src.data.pbr_estimation_dataset_2d import PBREstimationSample2D


@dataclass
class Prediction2D:
    """Predicted 2D material maps for one sample."""

    albedo: Image.Image
    roughness: Image.Image
    metallic: Image.Image

    def save(self, save_dir: Path, *, mark_success: bool = True) -> Prediction2D:
        """Save the predicted channels into ``save_dir`` and return self."""
        save_dir.mkdir(parents=True, exist_ok=True)

        self.albedo.save(save_dir / "albedo.png")
        self.roughness.save(save_dir / "roughness.png")
        self.metallic.save(save_dir / "metallic.png")

        if mark_success:
            (save_dir / ".SUCCESS").touch()

        return self




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

    @abstractmethod
    def predict(
        self,
        samples: Sequence[PBREstimationSample2D],
        output_dir: Path,
    ) -> Mapping[str, Prediction2D]:
        """Predict material maps for a non-empty collection of samples."""
