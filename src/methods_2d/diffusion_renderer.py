"""DiffusionRenderer screen-space material estimator.

DiffusionRenderer (Liang et al., 2025) estimates 2.5D/PBR attributes
(basecolor, roughness, metallic, normal, depth) from images/video sequences
using video diffusion models.
"""

from __future__ import annotations

import importlib
import logging
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
from PIL import Image
from tqdm.auto import tqdm

from src.data.pbr_estimation_dataset_2d import PBREstimationSample2D
from src.methods_2d.base import BaseMaterialEstimator2D, Prediction2D

logger = logging.getLogger(__name__)

_UPSTREAM_PACKAGE = "_pbr_eval_diffusion_renderer_upstream"
_GBUFFER_INDEX_MAPPING = {
    "basecolor": 0,
    "metallic": 1,
    "roughness": 2,
    "normal": 3,
}


def _import_upstream(repo_root: Path, module: str) -> ModuleType:
    """Import a DiffusionRenderer module without colliding with this project's ``src``."""
    upstream_src = str(repo_root / "src")
    upstream_root = str(repo_root)

    if upstream_root not in sys.path:
        sys.path.insert(0, upstream_root)

    package = sys.modules.get(_UPSTREAM_PACKAGE)
    if package is None:
        package = ModuleType(_UPSTREAM_PACKAGE)
        package.__package__ = _UPSTREAM_PACKAGE
        package.__path__ = [upstream_src]  # type: ignore[attr-defined]
        sys.modules[_UPSTREAM_PACKAGE] = package
    elif upstream_src not in package.__path__:  # type: ignore[attr-defined]
        raise ImportError(
            f"{_UPSTREAM_PACKAGE} points to a different DiffusionRenderer checkout"
        )
    return importlib.import_module(f"{_UPSTREAM_PACKAGE}.{module}")


class DiffusionRenderer2D(BaseMaterialEstimator2D):
    """DiffusionRenderer screen-space material estimator."""

    def __init__(
        self,
        *,
        config_path: str | Path = "third_party/diffusion-renderer/configs/rgbx_inference.yaml",
        checkpoint_dir: str | Path = "third_party/diffusion-renderer/checkpoints/diffusion_renderer-inverse-svd",
        device: str = "cuda:0",
        image_size: int = 512,
        n_steps: int = 20,
        dummy_roughness: float = 0.5,
        dummy_metallic: float = 0.0,
        seed: int = 0,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.config_path = self.resolve_path(config_path)
        self.checkpoint_dir = self.resolve_path(checkpoint_dir)
        self.device_str = device
        self.image_size = image_size
        self.n_steps = n_steps
        self.dummy_roughness = dummy_roughness
        self.dummy_metallic = dummy_metallic
        self.seed = seed
        self.pipeline: Any = None

    def setup(self) -> None:
        super().setup()

        if not self.checkpoint_dir.exists():
            raise FileNotFoundError(
                f"DiffusionRenderer checkpoint directory not found at: {self.checkpoint_dir}\n"
                f"Please download weights using:\n"
                f"third_party/.venvs/diffusion_renderer/bin/python "
                f"third_party/diffusion-renderer/utils/download_weights.py "
                f"--repo_id nexuslrf/diffusion_renderer-inverse-svd "
                f"--local_dir {self.checkpoint_dir}"
            )

        import torch
        from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection

        pipeline_module = _import_upstream(self.repo_root, "pipelines.pipeline_rgbx")
        RGBXVideoDiffusionPipeline = pipeline_module.RGBXVideoDiffusionPipeline

        missing_kwargs: dict[str, Any] = {
            "cond_mode": "skip",
            "use_deterministic_mode": False,
        }

        subfolders = [p.name for p in self.checkpoint_dir.iterdir() if p.is_dir()]
        if "image_encoder" not in subfolders:
            missing_kwargs["image_encoder"] = (
                CLIPVisionModelWithProjection.from_pretrained(
                    "stabilityai/stable-video-diffusion-img2vid",
                    subfolder="image_encoder",
                )
            )
        if "feature_extractor" not in subfolders:
            missing_kwargs["feature_extractor"] = (
                CLIPImageProcessor.from_pretrained(
                    "stabilityai/stable-video-diffusion-img2vid",
                    subfolder="feature_extractor",
                )
            )

        self.pipeline = RGBXVideoDiffusionPipeline.from_pretrained(
            str(self.checkpoint_dir), **missing_kwargs
        )
        self.pipeline = self.pipeline.to(self.device_str)
        self.pipeline = self.pipeline.to(torch.float16)
        self.pipeline.set_progress_bar_config(disable=True)

    def predict(
        self,
        samples: Sequence[PBREstimationSample2D],
        output_dir: Path,
    ) -> Mapping[str, Prediction2D]:
        if self.pipeline is None:
            raise RuntimeError("Method setup() was not called before predict().")

        import torch

        predictions: dict[str, Prediction2D] = {}
        device_type = torch.device(self.device_str).type

        for sample in tqdm(samples, desc=f"{self.name} 2D [{len(samples)} samples]"):
            sample_output_dir = output_dir / sample.sample_id

            # Load input RGB image
            with Image.open(sample.rgb) as input_image:
                rgb_pil = input_image.convert("RGB")
            if rgb_pil.size != (self.image_size, self.image_size):
                rgb_pil = rgb_pil.resize(
                    (self.image_size, self.image_size), resample=Image.BILINEAR
                )

            img_np = (
                np.asarray(rgb_pil).astype(np.float32) / 255.0
            )  # (H, W, 3)
            # DiffusionRenderer video input format: (1, F, H, W, C)
            input_images = img_np[None, None, ...]  # (1, 1, H, W, 3)

            cond_images = {"rgb": input_images}
            cond_labels = {"rgb": "vae"}

            results: dict[str, Image.Image] = {}
            for inference_pass in ["basecolor", "roughness", "metallic", "normal"]:
                # Passing the numeric context avoids upstream's ambiguous
                # absolute import from its repository-level ``utils`` folder.
                cond_images["input_context"] = torch.tensor(
                    [_GBUFFER_INDEX_MAPPING[inference_pass]],
                    device=self.device_str,
                    dtype=torch.long,
                )
                # Match upstream inference: each material pass starts from the
                # same seeded noise rather than advancing one shared generator.
                generator = (
                    torch.Generator(device=self.device_str).manual_seed(self.seed)
                    if self.seed is not None
                    else None
                )
                with torch.autocast(device_type, enabled=device_type == "cuda"):
                    out_frames = self.pipeline(
                        cond_images,
                        cond_labels,
                        height=self.image_size,
                        width=self.image_size,
                        num_frames=1,
                        num_inference_steps=self.n_steps,
                        min_guidance_scale=1.0,
                        max_guidance_scale=1.0,
                        noise_aug_strength=0.0,
                        generator=generator,
                    ).frames[0]
                    results[inference_pass] = out_frames[0]

            albedo_img = results.get("basecolor", rgb_pil)
            roughness_img = results.get("roughness")
            metallic_img = results.get("metallic")
            normal_img = results.get("normal")

            if roughness_img is None:
                dummy_r = (
                    np.full(
                        (self.image_size, self.image_size),
                        int(self.dummy_roughness * 255),
                        dtype=np.uint8,
                    )
                )
                roughness_img = Image.fromarray(dummy_r)

            if metallic_img is None:
                dummy_m = (
                    np.full(
                        (self.image_size, self.image_size),
                        int(self.dummy_metallic * 255),
                        dtype=np.uint8,
                    )
                )
                metallic_img = Image.fromarray(dummy_m)

            prediction = self.save_prediction(
                sample_output_dir,
                albedo=albedo_img,
                roughness=roughness_img,
                metallic=metallic_img,
                normal=normal_img,
            )
            predictions[sample.sample_id] = prediction

        return predictions
