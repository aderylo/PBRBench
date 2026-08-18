"""Base contract for 3D material estimators."""

from __future__ import annotations

import shutil
from abc import ABC, abstractmethod
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from PIL import Image

from src.data.pbr_estimation_dataset_3d import PBREstimationSample3D
from src.utils.glb import (
    create_textured_glb,
    extract_pbr_textures,
    remap_uv_textures,
)


@dataclass
class Prediction3D:
    """One self-contained 3D PBR material prediction."""

    sample_id: str
    pbr_asset_glb: Path | None = None
    albedo: Image.Image | Path | None = None
    roughness: Image.Image | Path | None = None
    metallic: Image.Image | Path | None = None

    def save(self, save_dir: Path, *, mark_success: bool = False) -> Prediction3D:
        """Save predicted material maps into ``save_dir`` and optionally mark success."""
        save_dir.mkdir(parents=True, exist_ok=True)
        if isinstance(self.albedo, Image.Image):
            self.albedo.save(save_dir / "albedo.png")
        if isinstance(self.roughness, Image.Image):
            self.roughness.save(save_dir / "roughness.png")
        if isinstance(self.metallic, Image.Image):
            self.metallic.save(save_dir / "metallic.png")

        if mark_success:
            (save_dir / ".SUCCESS").touch()

        return self

    @classmethod
    def from_dir(cls, sample_dir: Path) -> Prediction3D:
        """Construct Prediction3D from a prediction sample directory."""
        canonical_glb = sample_dir / "canonical_asset.glb"
        mesh_glb = sample_dir / "mesh.glb"
        asset_glb = (
            canonical_glb
            if canonical_glb.is_file()
            else (mesh_glb if mesh_glb.is_file() else None)
        )

        albedo_path = sample_dir / "albedo.png"
        roughness_path = sample_dir / "roughness.png"
        metallic_path = sample_dir / "metallic.png"

        return cls(
            sample_id=sample_dir.name,
            pbr_asset_glb=asset_glb,
            albedo=albedo_path if albedo_path.is_file() else None,
            roughness=roughness_path if roughness_path.is_file() else None,
            metallic=metallic_path if metallic_path.is_file() else None,
        )


class BaseMaterialEstimator3D(ABC):
    """Common dataset-level interface implemented by every 3D material estimator."""

    uv_correspondence: Literal["auto", "topology", "identity"] | None = "auto"
    texture_size: int = 1024

    def __init__(
        self,
        *,
        name: str = "estimator_3d",
        project_root: str | Path = ".",
        repo_root: str | Path = ".",
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
        """Optional hook: load upstream code and model weights before predicting."""

    def teardown(self) -> None:
        """Optional hook: release estimator-owned resources after predicting."""

    def align_to_original_uv(
        self,
        sample: PBREstimationSample3D,
        prediction: Prediction3D,
        output_dir: str | Path,
        *,
        debug: bool = False,
    ) -> Path:
        """Align predicted material textures to the original benchmark UV layout.

        Saves ``canonical_asset.glb`` into ``output_dir``. If ``uv_correspondence``
        is ``"identity"`` (or ``None``), remapping is skipped and the native GLB
        is copied directly.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        canonical_path = output_dir / "canonical_asset.glb"
        source_glb = Path(prediction.pbr_asset_glb).resolve()

        if self.uv_correspondence in (None, "identity"):
            if canonical_path.resolve() != source_glb:
                shutil.copy2(source_glb, canonical_path)
            if debug:
                native = extract_pbr_textures(source_glb)
                for layout in ("native", "canonical"):
                    debug_dir = output_dir / "debug" / layout
                    debug_dir.mkdir(parents=True, exist_ok=True)
                    for channel, image in native.items():
                        image.save(debug_dir / f"{channel}.png")
            return canonical_path

        transfer = remap_uv_textures(
            source_glb,
            sample.mesh_path,
            resolution=getattr(self, "texture_size", getattr(self, "uv_size", 1024)),
            correspondence=self.uv_correspondence,
        )
        canonical_asset = create_textured_glb(
            sample.mesh_path,
            transfer.textures,
            canonical_path,
        )
        if debug:
            native = extract_pbr_textures(source_glb)
            for layout, textures in (
                ("native", native),
                ("canonical", transfer.textures),
            ):
                debug_dir = output_dir / "debug" / layout
                debug_dir.mkdir(parents=True, exist_ok=True)
                for channel, image in textures.items():
                    image.save(debug_dir / f"{channel}.png")
            transfer.uv_mask.save(output_dir / "debug" / "canonical" / "uv_mask.png")

        return canonical_asset

    @abstractmethod
    def predict_over_dataset(
        self,
        samples: Sequence[PBREstimationSample3D],
        output_dir: str | Path,
    ) -> Iterator[Prediction3D]:
        """Predict 3D PBR material assets for a collection of samples."""
