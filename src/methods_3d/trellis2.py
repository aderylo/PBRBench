"""TRELLIS 2 single-image 3D material estimation adapter.

This adapter integrates the texture recovery pipeline of TRELLIS 2
(``Trellis2TexturingPipeline``) into the 3D PBR material estimation benchmark.
Given a 3D geometry mesh and its baked UV texture atlas, the adapter first
renders the mesh into a canonical 2D reference view using PyTorch3D and then
lets TRELLIS 2 decode a 3D sparse PBR voxel grid, which is baked into UV
texture maps containing Albedo (Base Color), Perceptual Roughness, and
Metallic channels.
"""

from __future__ import annotations

import gc
import logging
import sys
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
import torch
from tqdm.auto import tqdm

from src.data.pbr_estimation_dataset_3d import PBREstimationSample3D
from src.methods_3d.base import BaseMaterialEstimator3D, Prediction3D
from src.utils.rendering import render_reference_image

logger = logging.getLogger(__name__)


def _import_upstream(repo_root: Path) -> None:
    """Prepend upstream TRELLIS 2 repo to sys.path so its modules are importable."""
    repo_path = str(repo_root)
    if repo_path not in sys.path:
        sys.path.insert(0, repo_path)


class Trellis2Estimator3D(BaseMaterialEstimator3D):
    """TRELLIS 2 single-image 3D material estimation adapter."""

    uv_correspondence = "topology"

    def __init__(
        self,
        *,
        name: str = "trellis2",
        project_root: str | Path = ".",
        repo_root: str | Path = "third_party/TRELLIS.2",
        model_name_or_path: str = "microsoft/TRELLIS.2-4B",
        config_file: str = "texturing_pipeline.json",
        resolution: int = 1024,
        texture_size: int = 2048,
        seed: int = 42,
        tex_slat_sampling_steps: int = 12,
        tex_slat_guidance_strength: float = 1.0,
        tex_slat_guidance_rescale: float = 0.0,
        tex_slat_rescale_t: float = 3.0,
        camera_distance: float = 1.0,
        camera_elevation: float = 20.0,
        camera_azimuth: float = 0.0,
        camera_focal_length: float = 2.0,
        reference_resolution: int = 1024,
        device: str = "cuda",
        **kwargs: Any,
    ) -> None:
        super().__init__(name=name, project_root=project_root, repo_root=repo_root)
        self.model_name_or_path = model_name_or_path
        model_path = Path(model_name_or_path)
        local_candidate = self.project_root / model_path
        if not model_path.is_absolute() and local_candidate.exists():
            self.model_name_or_path = str(local_candidate.resolve())
        self.config_file = config_file
        self.resolution = int(resolution)
        self.texture_size = int(texture_size)
        self.seed = int(seed)
        self.tex_slat_sampling_steps = int(tex_slat_sampling_steps)
        self.tex_slat_guidance_strength = float(tex_slat_guidance_strength)
        self.tex_slat_guidance_rescale = float(tex_slat_guidance_rescale)
        self.tex_slat_rescale_t = float(tex_slat_rescale_t)
        self.camera_distance = float(camera_distance)
        self.camera_elevation = float(camera_elevation)
        self.camera_azimuth = float(camera_azimuth)
        self.camera_focal_length = float(camera_focal_length)
        self.reference_resolution = int(reference_resolution)
        self.device = device

        self._pipeline: Any = None

    def setup(self) -> None:
        """Initialize and load TRELLIS 2 texturing pipeline."""
        super().setup()
        _import_upstream(self.repo_root)
        from trellis2.pipelines.trellis2_texturing import (
            Trellis2TexturingPipeline,
        )

        logger.info(
            f"Loading TRELLIS 2 Texturing Pipeline from {self.model_name_or_path}..."
        )
        self._pipeline = Trellis2TexturingPipeline.from_pretrained(
            self.model_name_or_path,
            config_file=self.config_file,
        )
        self._pipeline.to(self.device)
        logger.info(f"TRELLIS 2 pipeline successfully loaded onto {self.device}.")

    def teardown(self) -> None:
        """Release GPU memory and pipeline resources."""
        if self._pipeline is not None:
            del self._pipeline
            self._pipeline = None

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
        gc.collect()
        logger.info("TRELLIS 2 estimator resources released.")

    def _extract_material_maps(
        self, textured_mesh: Any
    ) -> tuple[Image.Image, Image.Image, Image.Image]:
        """Extract Albedo, Roughness, and Metallic maps from trimesh PBRMaterial.

        TRELLIS 2 produces a trimesh with PBRMaterial containing:
        - `baseColorTexture`: PIL Image (RGB)
        - `metallicRoughnessTexture`: PIL Image where G=Roughness, B=Metallic
        """
        visual = getattr(textured_mesh, "visual", None)
        material = getattr(visual, "material", None) if visual else None

        if material is None:
            raise ValueError("Textured mesh does not contain valid material visuals")

        # 1. Extract Albedo / Base Color
        base_color = getattr(material, "baseColorTexture", None)
        if base_color is None:
            raise ValueError("Mesh material is missing baseColorTexture")
        if not isinstance(base_color, Image.Image):
            base_color = Image.fromarray(np.asarray(base_color))
        albedo_img = base_color.convert("RGB")

        # 2. Extract Metallic-Roughness texture
        mr_tex = getattr(material, "metallicRoughnessTexture", None)
        if mr_tex is not None:
            if not isinstance(mr_tex, Image.Image):
                mr_tex = Image.fromarray(np.asarray(mr_tex))
            mr_rgb = mr_tex.convert("RGB")
            # glTF standard: Green channel = Roughness, Blue channel = Metallic
            _, roughness_img, metallic_img = mr_rgb.split()
        else:
            # Fallback to scalar material properties if texture map is absent
            width, height = albedo_img.size
            roughness_val = int(
                getattr(material, "roughnessFactor", 0.5) * 255.0
            )
            metallic_val = int(
                getattr(material, "metallicFactor", 0.0) * 255.0
            )
            roughness_img = Image.new("L", (width, height), roughness_val)
            metallic_img = Image.new("L", (width, height), metallic_val)

        # Match dimensions to albedo if sizes differ
        if roughness_img.size != albedo_img.size:
            roughness_img = roughness_img.resize(
                albedo_img.size, Image.Resampling.BILINEAR
            )
        if metallic_img.size != albedo_img.size:
            metallic_img = metallic_img.resize(
                albedo_img.size, Image.Resampling.BILINEAR
            )

        return albedo_img, roughness_img, metallic_img

    def predict_over_dataset(
        self,
        samples: Sequence[PBREstimationSample3D],
        output_dir: str | Path,
    ) -> Iterator[Prediction3D]:
        """Predict 3D PBR material maps for a collection of 3D samples."""
        if self._pipeline is None:
            raise RuntimeError("Estimator is not set up. Call setup() first.")

        output_path = Path(output_dir)
        for sample in tqdm(samples, desc=f"Predicting with {self.name}"):
            yield self._predict_sample(sample, output_path / sample.sample_id)

    def _predict_sample(
        self, sample: PBREstimationSample3D, sample_dir: Path
    ) -> Prediction3D:
        """Predict 3D PBR material maps for a single 3D sample."""

        # 1. Load input mesh
        mesh = sample.load_trimesh(process=False)

        # 2. Use canonical reference view if available, or render via PyTorch3D
        if sample.reference_view is not None and sample.reference_view.is_file():
            ref_image = Image.open(sample.reference_view).convert("RGB")
            if not (sample_dir / "reference_image.png").is_file():
                sample_dir.mkdir(parents=True, exist_ok=True)
                ref_image.save(sample_dir / "reference_image.png")
        else:
            ref_image = render_reference_image(
                sample,
                camera_distance=self.camera_distance,
                camera_elevation=self.camera_elevation,
                camera_azimuth=self.camera_azimuth,
                camera_focal_length=self.camera_focal_length,
                resolution=self.reference_resolution,
                device=self.device,
                output_path=sample_dir / "reference_image.png",
            )

        # 3. Run TRELLIS 2 texturing pipeline
        sampler_params = {
            "steps": self.tex_slat_sampling_steps,
            "guidance_strength": self.tex_slat_guidance_strength,
            "guidance_rescale": self.tex_slat_guidance_rescale,
            "rescale_t": self.tex_slat_rescale_t,
        }

        out_mesh = self._pipeline.run(
            mesh,
            ref_image,
            seed=self.seed,
            tex_slat_sampler_params=sampler_params,
            preprocess_image=False,
            resolution=self.resolution,
            texture_size=self.texture_size,
        )

        # 4. Extract Albedo, Roughness, Metallic maps
        albedo_img, roughness_img, metallic_img = self._extract_material_maps(out_mesh)

        # 5. Persist prediction files and the textured mesh
        sample_dir.mkdir(parents=True, exist_ok=True)
        albedo_path = sample_dir / "albedo.png"
        roughness_path = sample_dir / "roughness.png"
        metallic_path = sample_dir / "metallic.png"
        mesh_path = sample_dir / "mesh.glb"

        albedo_img.save(albedo_path)
        roughness_img.save(roughness_path)
        metallic_img.save(metallic_path)
        out_mesh.export(str(mesh_path), extension_webp=False)

        return Prediction3D(
            sample_id=sample.sample_id,
            pbr_asset_glb=mesh_path,
        )

