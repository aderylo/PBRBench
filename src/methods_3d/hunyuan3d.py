"""Hunyuan3D-2.1 (Hunyuan3D-Paint) 3D material estimation adapter.

This adapter integrates the texture generation pipeline of Hunyuan3D-Paint 2.1
into the 3D PBR material estimation benchmark. Given a 3D geometry mesh and its
baked UV texture atlas, the adapter first renders the mesh into a canonical 2D
reference view using PyTorch3D and then lets Hunyuan3D-Paint 2.1 estimate the
multi-view PBR textures (Albedo and Metallic-Roughness) and back-project them onto
the mesh's UV map.
"""

from __future__ import annotations

import gc
import logging
import sys
import tempfile
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
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


@dataclass(frozen=True)
class _UpstreamModules:
    """Typed container holding imported upstream modules."""

    Hunyuan3DPaintConfig: Any
    Hunyuan3DPaintPipeline: Any


def _import_upstream(repo_root: Path) -> _UpstreamModules:
    """Import upstream Hunyuan3D modules safely."""
    repo_path = str(repo_root)
    hy3dpaint_path = str(repo_root / "hy3dpaint")

    if repo_path not in sys.path:
        sys.path.insert(0, repo_path)
    if hy3dpaint_path not in sys.path:
        sys.path.insert(0, hy3dpaint_path)

    try:
        from torchvision_fix import apply_fix

        apply_fix()
    except Exception:
        pass

    from textureGenPipeline import Hunyuan3DPaintConfig, Hunyuan3DPaintPipeline

    return _UpstreamModules(
        Hunyuan3DPaintConfig=Hunyuan3DPaintConfig,
        Hunyuan3DPaintPipeline=Hunyuan3DPaintPipeline,
    )


class Hunyuan3DEstimator3D(BaseMaterialEstimator3D):
    """Hunyuan3D-Paint 2.1 3D PBR material estimator adapter."""

    uv_correspondence = "auto"

    def __init__(
        self,
        *,
        name: str = "hunyuan3d",
        project_root: str | Path = ".",
        repo_root: str | Path = "third_party/Hunyuan3D-2.1",
        weights_dir: str | Path | None = None,
        max_num_view: int = 6,
        resolution: int = 512,
        texture_size: int = 2048,
        camera_distance: float = 1.6,
        camera_elevation: float = 0.0,
        camera_azimuth: float = 0.0,
        camera_focal_length: float = 1.375,
        reference_resolution: int = 512,
        device: str = "cuda",
        **kwargs: Any,
    ) -> None:
        super().__init__(name=name, project_root=project_root, repo_root=repo_root)
        self.weights_dir = self.resolve_path(weights_dir) if weights_dir else None
        self.max_num_view = int(max_num_view)
        self.resolution = int(resolution)
        self.texture_size = int(texture_size)
        self.camera_distance = float(camera_distance)
        self.camera_elevation = float(camera_elevation)
        self.camera_azimuth = float(camera_azimuth)
        self.camera_focal_length = float(camera_focal_length)
        self.reference_resolution = int(reference_resolution)
        self.device = device

        self._pipeline: Any = None
        self._up: _UpstreamModules | None = None

    def setup(self) -> None:
        """Initialize and load Hunyuan3D-Paint pipeline."""
        super().setup()
        self._up = _import_upstream(self.repo_root)

        logger.info("Initializing Hunyuan3D-Paint 2.1 Pipeline...")
        config = self._up.Hunyuan3DPaintConfig(
            max_num_view=self.max_num_view,
            resolution=self.resolution,
        )
        config.device = self.device
        config.texture_size = self.texture_size
        config.render_size = self.texture_size

        if self.weights_dir is None:
            default_weights = self.repo_root.parents[1] / ".weights" / "hunyuan3d"
            if default_weights.is_dir():
                self.weights_dir = default_weights

        if self.weights_dir is not None:
            paintpbr_dir = self.weights_dir / "hunyuan3d-paintpbr-v2-1"
            if paintpbr_dir.is_dir():
                config.multiview_pretrained_path = str(paintpbr_dir)
            else:
                config.multiview_pretrained_path = str(self.weights_dir)

            dino_dir = self.weights_dir / "dinov2-giant"
            if dino_dir.is_dir():
                config.dino_ckpt_path = str(dino_dir)

            realesrgan_path = self.weights_dir / "ckpt" / "RealESRGAN_x4plus.pth"
            if realesrgan_path.is_file():
                config.realesrgan_ckpt_path = str(realesrgan_path)

        cfg_path = self.repo_root / "hy3dpaint" / "cfgs" / "hunyuan-paint-pbr.yaml"
        if cfg_path.is_file():
            config.multiview_cfg_path = str(cfg_path)
        config.custom_pipeline = str(self.repo_root / "hy3dpaint" / "hunyuanpaintpbr")

        self._pipeline = self._up.Hunyuan3DPaintPipeline(config)
        logger.info("Hunyuan3D-Paint 2.1 pipeline successfully loaded.")

    def teardown(self) -> None:
        """Release GPU memory and pipeline resources."""
        if self._pipeline is not None:
            del self._pipeline
            self._pipeline = None
        self._up = None

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
        gc.collect()
        logger.info("Hunyuan3D estimator resources released.")

    def _predict_sample(
        self, sample: PBREstimationSample3D, sample_dir: Path
    ) -> Prediction3D:
        """Predict 3D PBR material maps for a single 3D sample."""
        # 1. Use canonical reference view if available, or render via PyTorch3D
        if sample.reference_view is not None and sample.reference_view.is_file():
            sample_dir.mkdir(parents=True, exist_ok=True)
            ref_image_path = sample_dir / "reference_image.png"
            if not ref_image_path.is_file():
                ref_img_pil = Image.open(sample.reference_view).convert("RGB")
                ref_img_pil.save(ref_image_path)
            ref_image = ref_image_path
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

        # 2. Run Hunyuan3D-Paint texturing pipeline
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_output_obj = Path(tmp_dir) / "output.obj"
            self._pipeline(
                mesh_path=str(sample.mesh_path),
                image_path=ref_image,
                output_mesh_path=str(tmp_output_obj),
                use_remesh=False,
                save_glb=True,
            )

            # Retrieve baked textures from renderer
            albedo_tex = self._pipeline.render.tex
            mr_tex = self._pipeline.render.tex_mr

            if isinstance(albedo_tex, torch.Tensor):
                albedo_np = (albedo_tex.cpu().numpy() * 255.0).round().clip(0, 255).astype(np.uint8)
            else:
                albedo_np = np.asarray(albedo_tex, dtype=np.uint8)

            if isinstance(mr_tex, torch.Tensor):
                mr_np = (mr_tex.cpu().numpy() * 255.0).round().clip(0, 255).astype(np.uint8)
            else:
                mr_np = np.asarray(mr_tex, dtype=np.uint8)

            # R channel: Metallic, G channel: Roughness
            metallic_np = mr_np[..., 0]
            roughness_np = mr_np[..., 1]

            albedo_img = Image.fromarray(albedo_np).convert("RGB")
            roughness_img = Image.fromarray(roughness_np, mode="L")
            metallic_img = Image.fromarray(metallic_np, mode="L")

            # 3. Persist prediction files
            sample_dir.mkdir(parents=True, exist_ok=True)
            albedo_path = sample_dir / "albedo.png"
            roughness_path = sample_dir / "roughness.png"
            metallic_path = sample_dir / "metallic.png"
            mesh_path = sample_dir / "mesh.glb"

            albedo_img.save(albedo_path)
            roughness_img.save(roughness_path)
            metallic_img.save(metallic_path)

            tmp_output_glb = tmp_output_obj.with_suffix(".glb")
            if tmp_output_glb.is_file():
                import shutil
                shutil.move(str(tmp_output_glb), str(mesh_path))

            return Prediction3D(
                sample_id=sample.sample_id,
                pbr_asset_glb=mesh_path,
            )

    def predict_over_dataset(
        self,
        samples: Sequence[PBREstimationSample3D],
        output_dir: str | Path,
    ) -> Iterator[Prediction3D]:
        """Predict 3D material maps for a collection of 3D samples."""
        if self._pipeline is None or self._up is None:
            raise RuntimeError("Estimator is not set up. Call setup() first.")

        output_path = Path(output_dir)
        for sample in tqdm(samples, desc=f"Predicting with {self.name}"):
            yield self._predict_sample(sample, output_path / sample.sample_id)
