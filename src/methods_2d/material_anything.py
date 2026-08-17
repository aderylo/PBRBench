"""Material Anything single-view material estimator (2D adapter).

Material Anything (Huang et al., 2024) is a multi-view PBR texture estimator.
Its image-to-materials stage is a conditional latent-diffusion UNet with three
parallel input heads (albedo / roughness-metallic / bump) sharing one
backbone. Each head receives its own 4 latent channels plus 9 conditioning
channels (4 image latents + 4 camera-space normal latents + 1 keep-mask).

The upstream multi-view loop calls this stage per view with progressively
projected init materials and a keep-mask. Its first view is effectively pure
single-view inference: white placeholder init materials and a keep-mask that
only covers the background. This adapter reproduces that first-view call with
the benchmark's registered RGB, camera-space normal, and foreground mask, so
the estimator can be benchmarked against the other screen-space methods.

Convention notes:
- The upstream estimator conditions on camera-space normals rendered by
  PyTorch3D and converted with the LUB2RUF transform (front-facing = +Z,
  white background). The benchmark's ``normal.png`` is Blender camera-space
  (front-facing = -Z, black background), so the blue channel is flipped and
  the background is filled with white before inference.
- The roughness-metallic output encodes roughness in G and metallic in B
  (the saturated R channel is discarded, matching upstream).
- The estimator does not predict normals (they are an input conditioning);
  its third head output is a bump/displacement-style map and is saved as an
  artifact named ``bump``, not as a normal prediction.
"""

from __future__ import annotations

import logging
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm.auto import tqdm

from src.data.pbr_estimation_dataset_2d import PBREstimationSample2D
from src.methods_2d.base import BaseMaterialEstimator2D, Prediction2D

logger = logging.getLogger(__name__)


class MaterialAnything2D(BaseMaterialEstimator2D):
    def __init__(
        self,
        *,
        name: str = "material_anything",
        project_root: str | Path,
        repo_root: str | Path = "third_party/MaterialAnything",
        image2materials_model: str | Path = ".weights/material_anything/material_estimator",
        device: str = "cuda:0",
        image_size: int = 768,
        num_inference_steps: int = 50,
        seed: int = 0,
        **kwargs,
    ) -> None:
        super().__init__(
            name=name, project_root=project_root, repo_root=repo_root
        )
        model_path = Path(image2materials_model)
        if model_path.is_absolute():
            self.image2materials_model = model_path.resolve()
        else:
            project_candidate = (self.project_root / model_path).resolve()
            self.image2materials_model = (
                project_candidate
                if project_candidate.exists()
                else (self.repo_root / model_path).resolve()
            )
        self.device = str(device)
        self.image_size = int(image_size)
        self.num_inference_steps = int(num_inference_steps)
        self.seed = int(seed)
        self.pipeline = None

    def _import_upstream(self):
        """Import the upstream pipeline and scheduler without ControlNet code."""
        repo_path = str(self.repo_root)
        if repo_path not in sys.path:
            sys.path.insert(0, repo_path)

        import torch
        from models.scheduling_ddpm import DDPMScheduler
        from pipelines.pipeline_stable_diffusion_switcher import (
            StableDiffusionPipeline as MaterialEstimatorPipeline,
        )

        return torch, MaterialEstimatorPipeline, DDPMScheduler

    def setup(self) -> None:
        super().setup()
        if not self.image2materials_model.is_dir():
            raise FileNotFoundError(
                "Material Anything estimator model not found at "
                f"{self.image2materials_model}. Run "
                "third_party/MaterialAnything/download_models.sh."
            )

        torch, Pipeline, DDPMScheduler = self._import_upstream()
        logger.info("Loading Material Anything estimator into GPU VRAM.")
        self.pipeline = Pipeline.from_pretrained(
            str(self.image2materials_model), torch_dtype=torch.float16
        ).to(self.device)
        self.pipeline.scheduler = DDPMScheduler.from_pretrained(
            str(self.image2materials_model), subfolder="scheduler"
        )

    def teardown(self) -> None:
        self.pipeline = None
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _load_condition_image(self, sample: PBREstimationSample2D) -> Image.Image:
        image = Image.open(sample.rgb).convert("RGB")
        return image.resize(
            (self.image_size, self.image_size), Image.Resampling.BILINEAR
        )

    def _load_condition_normal(self, sample: PBREstimationSample2D) -> Image.Image:
        """Convert benchmark normals to the upstream camera-space convention."""
        if sample.normal is None or not sample.normal.is_file():
            raise FileNotFoundError(
                f"Material Anything 2D requires a registered camera-space "
                f"normal map, but none was found for {sample.sample_id}."
            )

        arr = np.array(Image.open(sample.normal).convert("RGB"), dtype=np.uint8)
        arr[..., 2] = 255 - arr[..., 2]
        if sample.mask is not None and sample.mask.is_file():
            mask = np.asarray(
                Image.open(sample.mask).convert("L"), dtype=np.float32
            ) / 255.0
            mask = mask[..., None]
            arr = (arr.astype(np.float32) * mask + 255.0 * (1.0 - mask)).astype(
                np.uint8
            )
        image = Image.fromarray(arr, "RGB")
        return image.resize(
            (self.image_size, self.image_size), Image.Resampling.BILINEAR
        )

    def _keep_mask(self, sample: PBREstimationSample2D, device):
        """Return the background keep-mask as a [1, H, W] tensor."""
        import torch

        height = width = self.image_size
        if sample.mask is not None and sample.mask.is_file():
            mask = Image.open(sample.mask).convert("L")
            mask = mask.resize((width, height), Image.Resampling.NEAREST)
            mask = torch.from_numpy(
                np.asarray(mask, dtype=np.float32) / 255.0
            ).to(device)
            return (1.0 - mask).unsqueeze(0)
        return torch.zeros((1, height, width), device=device)

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

        device = torch.device(self.device)
        generator = torch.Generator(device=self.device).manual_seed(self.seed)
        height = width = self.image_size

        outputs = {}
        for sample in tqdm(
            samples,
            desc=f"Material Anything 2D [{len(samples)} samples]",
            unit="sample",
        ):
            sample_dir = output_dir / sample.sample_id

            with Image.open(sample.rgb) as source:
                native_size = source.size

            cond_image = self._load_condition_image(sample)
            normal_image = self._load_condition_normal(sample)
            keep_mask = self._keep_mask(sample, device)
            init_materials = {
                "albedo": torch.ones(1, height, width, 3, device=device),
                "roughness_metallic": torch.ones(1, height, width, 3, device=device),
                "bump": torch.ones(1, height, width, 3, device=device),
            }

            albedo, rm, bump = self.pipeline(
                prompt=[""],
                cond_image=[cond_image],
                normal_image=[normal_image],
                init_materials=init_materials,
                masks=keep_mask,
                num_inference_steps=self.num_inference_steps,
                guidance_scale=1.0,
                generator=generator,
                height=height,
                width=width,
            ).images

            albedo = albedo.resize(native_size, Image.Resampling.BILINEAR)
            rm = rm.resize(native_size, Image.Resampling.BILINEAR)
            _, roughness, metallic = rm.convert("RGB").split()

            outputs[sample.sample_id] = Prediction2D(
                albedo=albedo,
                roughness=roughness,
                metallic=metallic,
            ).save(save_dir=sample_dir, mark_success=True)


        return outputs
