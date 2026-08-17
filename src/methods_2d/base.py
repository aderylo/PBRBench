"""Base contract and standard utilities for screen-space material estimators."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from src.data.pbr_estimation_dataset_2d import PBREstimationSample2D
from src.utils.image import save_image


@dataclass
class Prediction2D:
    """Predicted 2D material maps for one sample.

    Channels hold raw inputs (images/tensors/paths) until :meth:`save` is
    called, which replaces them with the paths of the saved files.
    """

    albedo: ImageInput
    roughness: ImageInput
    metallic: ImageInput
    artifacts: Mapping[str, ImageInput] = field(default_factory=dict)

    def save(self, save_dir: Path, *, mark_success: bool = True) -> Prediction2D:
        """Save the predicted channels into ``save_dir`` and return self."""
        save_dir.mkdir(parents=True, exist_ok=True)

        self.albedo = save_image(self.albedo, save_dir / "albedo.png")
        self.roughness = save_image(self.roughness, save_dir / "roughness.png")
        self.metallic = save_image(self.metallic, save_dir / "metallic.png")

        saved_artifacts: dict[str, Path] = {}
        for name, art_input in self.artifacts.items():
            if name in ("albedo", "roughness", "metallic"):
                continue
            ext = (
                ".png"
                if not isinstance(art_input, (str, Path))
                or not Path(art_input).suffix
                else Path(art_input).suffix
            )
            saved_artifacts[name] = save_image(
                art_input, save_dir / f"{name}{ext}"
            )
        self.artifacts = saved_artifacts

        if mark_success:
            (save_dir / ".SUCCESS").touch()

        return self


ImageInput = Any  # Image.Image | np.ndarray | torch.Tensor | Path


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
