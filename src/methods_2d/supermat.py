"""SuperMat screen-space material estimator.

SuperMat (Hong et al., ICCV 2025) estimates physically consistent PBR material
maps (albedo, roughness, metallic) from a single monocular RGBA image.

The single-image estimation workflow operates as follows:
1. Input Preprocessing: Composites the RGBA input image against a neutral gray
   background (0.5, 0.5, 0.5) and scales RGB intensities to [-1, 1].
2. Latent Diffusion Sampling: Uses a fine-tuned Stable Diffusion 2.1 UNet backbone
   with single-step (or multi-step) DDIM sampling, replicating target latents for
   Albedo and ORM (Occlusion, Roughness, Metallic) outputs.
3. Feature Decoding: Decodes predicted latents through the VAE decoder to extract
   the base color (Albedo) image and scalar Roughness (G channel) and Metallic (B channel)
   maps.
"""

from __future__ import annotations

import importlib
import logging
import random
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from types import ModuleType

from PIL import Image
from tqdm.auto import tqdm

from src.data.pbr_estimation_dataset_2d import PBREstimationSample2D
from src.methods_2d.base import BaseMaterialEstimator2D, Prediction2D

logger = logging.getLogger(__name__)

_UPSTREAM_PACKAGE = "_pbr_eval_supermat_upstream"


def _import_upstream(repo_root: Path, module: str) -> ModuleType:
    """Import a SuperMat module without colliding with this project's ``src``."""
    upstream_src = str(repo_root / "src")
    package = sys.modules.get(_UPSTREAM_PACKAGE)
    if package is None:
        package = ModuleType(_UPSTREAM_PACKAGE)
        package.__package__ = _UPSTREAM_PACKAGE
        package.__path__ = [upstream_src]  # type: ignore[attr-defined]
        sys.modules[_UPSTREAM_PACKAGE] = package
    elif upstream_src not in package.__path__:  # type: ignore[attr-defined]
        raise ImportError(
            f"{_UPSTREAM_PACKAGE} already points to a different SuperMat checkout"
        )
    return importlib.import_module(f"{_UPSTREAM_PACKAGE}.{module}")


class SuperMat2D(BaseMaterialEstimator2D):
    def __init__(
        self,
        *,
        checkpoint: str | Path,
        base_model: str = "sd2-community/stable-diffusion-2-1",
        device: str = "cuda:0",
        image_size: int = 512,
        inference_steps: int = 1,
        seed: int | None = 0,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.checkpoint = self.resolve_path(checkpoint)
        self.base_model = str(base_model)
        self.device = str(device)
        self.image_size = int(image_size)
        self.inference_steps = int(inference_steps)
        self.seed = seed
        self.pipeline = None

    def setup(self) -> None:
        super().setup()
        self.require_file(self.checkpoint, "SuperMat checkpoint")

        import numpy as np
        import torch
        from diffusers import DDIMScheduler

        adapters = _import_upstream(self.repo_root, "adapters")
        pipeline_module = _import_upstream(
            self.repo_root, "pipelines.pipeline_supermat_stable_diffusion"
        )
        upstream_utils = _import_upstream(self.repo_root, "utils")
        SuperMatAdapterWrapper = adapters.SuperMatAdapterWrapper
        SuperMatStableDiffusionPipeline = (
            pipeline_module.SuperMatStableDiffusionPipeline
        )

        if self.seed is not None:
            random.seed(self.seed)
            np.random.seed(self.seed)
            torch.manual_seed(self.seed)

        logger.info("Loading SuperMat base pipeline from '%s'...", self.base_model)
        pipe = SuperMatStableDiffusionPipeline.from_pretrained(
            self.base_model,
            safety_checker=None,
            requires_safety_checker=False,
        )

        pipe = SuperMatAdapterWrapper.convert(
            pipe,
            use_camera_embeddings=False,
            camera_embeddings_dim=16,
        )

        logger.info("Loading SuperMat UNet weights from '%s'...", self.checkpoint)
        unet_weights = upstream_utils.load_unet_weights(self.checkpoint)
        pipe.unet.load_state_dict(unet_weights, strict=False)
        pipe.unet.eval()

        pipe.scheduler = DDIMScheduler.from_config(
            pipe.scheduler.config, timestep_spacing="trailing"
        )
        self.pipeline = pipe.to(self.device)

    def teardown(self) -> None:
        import torch

        self.pipeline = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def predict(
        self,
        samples: Sequence[PBREstimationSample2D],
        output_dir: Path,
    ) -> Mapping[str, Prediction2D]:
        if self.pipeline is None:
            raise RuntimeError("Call setup() before predict()")

        if not samples:
            return {}

        import torch

        upstream_utils = _import_upstream(self.repo_root, "utils")

        torch_device = torch.device(self.device)

        generator = None
        if self.seed is not None:
            generator = torch.Generator(device=torch_device.type)
            generator.manual_seed(self.seed)

        outputs = {}
        for sample in tqdm(
            samples,
            desc=f"SuperMat 2D [{len(samples)} samples]",
            unit="sample",
        ):
            sample_dir = output_dir / sample.sample_id
            sample_dir.mkdir(parents=True, exist_ok=True)

            image_tensor = upstream_utils.load_rgba_image_as_rgb_tensor(
                image_path=sample.rgb,
                image_size=self.image_size,
                device=torch_device,
            )

            with torch.no_grad():
                images = self.pipeline(
                    prompt="",
                    num_inference_steps=self.inference_steps,
                    source_image=image_tensor,
                    output_type="pt",
                    generator=generator,
                )

            albedo_np = upstream_utils.to_uint8_rgb(images[0])
            roughness_np, metallic_np = upstream_utils.orm_to_roughness_metallic(
                images[1]
            )

            albedo_img = Image.fromarray(albedo_np)
            roughness_img = Image.fromarray(roughness_np)
            metallic_img = Image.fromarray(metallic_np)

            outputs[sample.sample_id] = self.save_prediction(
                sample_dir=sample_dir,
                albedo=albedo_img,
                roughness=roughness_img,
                metallic=metallic_img,
            )

        return outputs
