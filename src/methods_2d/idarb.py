"""IDArb screen-space material estimator.

IDArb (Li et al., ICLR 2025) performs intrinsic decomposition with a
Stable Diffusion 2.1 based multi-task latent diffusion prior that predicts
albedo, normal, and metallic/roughness (mro) domains for an arbitrary number
of input views. This adapter uses the released single-image workflow.

The single-image estimation workflow operates as follows:
1. Input Preprocessing: Resizes the observation so its longer side is 512
   pixels, pads it to a 512x512 canvas, and composites the foreground over a
   white background, matching the upstream CustomDataset.
2. Latent Diffusion Sampling: Runs the IDArbDiffusionPipeline with three task
   prompts ("albedo", "normal", "metallic and roughness") and 50 DDIM steps.
3. Channel Decoding: Splits the decoded output into the albedo RGB image, the
   normal RGB image, and the mro image whose channels 0 and 1 carry metallic
   and roughness. Predictions are cropped back and resized to the original
   observation resolution.
"""

from __future__ import annotations

import gc
import logging
import math
import random
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
from tqdm.auto import tqdm

from src.data.pbr_estimation_dataset_2d import PBREstimationSample2D
from src.methods_2d.base import BaseMaterialEstimator2D, Prediction2D

logger = logging.getLogger(__name__)

_DIFFUSION_SIZE = 512
_NUM_DOMAINS = 3
_MODEL_SUBFOLDERS = (
    "unet",
    "vae",
    "text_encoder",
    "tokenizer",
    "feature_extractor",
    "scheduler",
)


def _import_upstream(repo_root: Path) -> tuple[Any, Any]:
    """Import the upstream UNet and pipeline classes from the pinned checkout."""
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    existing = sys.modules.get("idarbdiffusion")
    if existing is not None:
        existing_paths = tuple(getattr(existing, "__path__", ()))
        if str(repo_root) not in existing_paths:
            raise ImportError(
                "A different top-level idarbdiffusion package is already loaded; "
                "IDArb cannot be imported safely in this process."
            )

    try:
        from idarbdiffusion.models.unet_dr2d_condition import (
            UNetDR2DConditionModel,
        )
        from idarbdiffusion.pipelines.pipeline_idarbdiffusion import (
            IDArbDiffusionPipeline,
        )
    except ModuleNotFoundError as error:
        raise ModuleNotFoundError(
            "IDArb dependencies are missing. Run "
            "scripts/setup/idarb_deps.py in its method environment."
        ) from error
    return UNetDR2DConditionModel, IDArbDiffusionPipeline


def _load_alpha(sample: PBREstimationSample2D) -> Image.Image | None:
    """Load the foreground alpha channel for a sample, if available."""
    if sample.mask is None:
        return None

    with Image.open(sample.mask) as mask_file:
        if mask_file.mode in {"1", "L", "I", "F"}:
            return mask_file.convert("L")
        if "A" in mask_file.getbands():
            return mask_file.getchannel("A")
        return mask_file.convert("L")


class IDArb2D(BaseMaterialEstimator2D):
    """Run IDArb's single-image intrinsic decomposition."""

    def __init__(
        self,
        *,
        model_dir: str | Path = ".weights/idarb",
        device: str = "cuda:0",
        dtype: str = "float32",
        num_inference_steps: int = 50,
        guidance_scale: float = 1.0,
        eta: float = 1.0,
        batch_size: int = 4,
        seed: int | None = 0,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.model_dir = self.resolve_path(model_dir)
        self.device = str(device)
        self.dtype = str(dtype)
        self.num_inference_steps = int(num_inference_steps)
        self.guidance_scale = float(guidance_scale)
        self.eta = float(eta)
        self.batch_size = int(batch_size)
        self.seed = seed
        self.pipeline: Any = None

    def setup(self) -> None:
        super().setup()

        if self.num_inference_steps <= 0:
            raise ValueError("num_inference_steps must be positive")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if self.dtype not in {"float32", "float16"}:
            raise ValueError("dtype must be 'float32' or 'float16'")

        for subfolder in _MODEL_SUBFOLDERS:
            if not (self.model_dir / subfolder).is_dir():
                raise FileNotFoundError(
                    f"IDArb model subfolder not found: {self.model_dir / subfolder}"
                )

        if self.device.startswith("cuda"):
            import torch

            if not torch.cuda.is_available():
                raise RuntimeError(
                    f"IDArb is configured for {self.device}, but CUDA is unavailable"
                )

        self._seed_everything()
        self._setup_pipeline()

    def _setup_pipeline(self) -> None:
        import torch
        from diffusers import AutoencoderKL, DDIMScheduler
        from transformers import CLIPImageProcessor, CLIPTextModel, CLIPTokenizer

        unet_cls, pipeline_cls = _import_upstream(self.repo_root)
        torch_dtype = torch.float16 if self.dtype == "float16" else torch.float32

        logger.info("Loading IDArb pipeline from '%s'...", self.model_dir)
        text_encoder = CLIPTextModel.from_pretrained(
            self.model_dir, subfolder="text_encoder", torch_dtype=torch_dtype
        )
        tokenizer = CLIPTokenizer.from_pretrained(self.model_dir, subfolder="tokenizer")
        feature_extractor = CLIPImageProcessor.from_pretrained(
            self.model_dir, subfolder="feature_extractor"
        )
        vae = AutoencoderKL.from_pretrained(
            self.model_dir, subfolder="vae", torch_dtype=torch_dtype
        )
        scheduler = DDIMScheduler.from_pretrained(self.model_dir, subfolder="scheduler")
        unet = unet_cls.from_pretrained(
            self.model_dir, subfolder="unet", torch_dtype=torch_dtype
        )

        try:
            self.pipeline = pipeline_cls(
                text_encoder=text_encoder,
                tokenizer=tokenizer,
                feature_extractor=feature_extractor,
                vae=vae,
                unet=unet,
                safety_checker=None,
                scheduler=scheduler,
            )
            # IDArb's joint-domain attention processor calls xformers directly;
            # without it the default processors silently ignore the CaPE pose
            # and domain-joining arguments and produce wrong predictions.
            try:
                from diffusers.utils.import_utils import is_xformers_available

                if not is_xformers_available():
                    raise RuntimeError(
                        "xformers is required by IDArb's joint-domain attention. "
                        "Install it in the idarb environment."
                    )
            except ImportError as error:
                raise RuntimeError(
                    "xformers is required by IDArb's joint-domain attention. "
                    "Install it in the idarb environment."
                ) from error
            self.pipeline.unet.enable_xformers_memory_efficient_attention()
            self.pipeline.to(self.device)
        except Exception:
            self.pipeline = None
            self._empty_cuda_cache()
            raise

    def teardown(self) -> None:
        self.pipeline = None
        self._empty_cuda_cache()

    def _empty_cuda_cache(self) -> None:
        try:
            import torch

            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass

    def _seed_everything(self) -> None:
        if self.seed is None:
            return

        random.seed(self.seed)
        np.random.seed(self.seed)
        import torch

        torch.manual_seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed)

    def _prepare_sample(self, sample: PBREstimationSample2D) -> dict[str, Any]:
        with Image.open(sample.rgb) as rgb_file:
            rgb_image = rgb_file.convert("RGB")
            original_size = rgb_image.size
            if sample.mask is None and "A" in rgb_file.getbands():
                alpha_image = rgb_file.getchannel("A")
            else:
                alpha_image = _load_alpha(sample)

        rgba = rgb_image.copy()
        if alpha_image is not None:
            rgba.putalpha(alpha_image)

        # Match the released preprocessing: resize the longer side to 512,
        # zero-pad to a 512x512 canvas, and composite the foreground over
        # white. The padding is zero/transparent, which composites to white.
        max_length = _DIFFUSION_SIZE
        width, height = rgba.size
        if width > height:
            resized_w = max_length
            resized_h = round(height / width * max_length)
        else:
            resized_h = max_length
            resized_w = round(width / height * max_length)
        resized = rgba.resize((resized_w, resized_h), Image.Resampling.BICUBIC)

        pad_left = (max_length - resized_w) // 2
        pad_top = (max_length - resized_h) // 2

        resized_rgb = np.asarray(resized.convert("RGB"), dtype=np.float32) / 255.0
        resized_alpha = (
            np.asarray(resized.getchannel("A"), dtype=np.float32) / 255.0 > 0.5
        )
        composite = resized_rgb * resized_alpha[..., None] + 1.0 * (
            1 - resized_alpha[..., None]
        )

        canvas = np.ones((max_length, max_length, 3), dtype=np.float32)
        canvas[pad_top : pad_top + resized_h, pad_left : pad_left + resized_w] = (
            composite
        )

        import torch

        # Camera pose encoding replicated from the upstream CustomDataset.
        theta = (
            (math.log(1.5) - math.log(1.2)) / (math.log(2.2) - math.log(1.2))
        ) * math.pi
        pose = torch.tensor([[0.0, 0.0, theta, 0.0]], dtype=torch.float32)

        return {
            "sample": sample,
            "img": torch.from_numpy(np.ascontiguousarray(canvas)).permute(2, 0, 1),
            "original_size": original_size,
            "crop_box": (pad_top, pad_left, resized_h, resized_w),
            "pose": pose,
        }

    @staticmethod
    def _restore_channel(
        prediction: Any,
        crop_box: tuple[int, int, int, int],
        original_size: tuple[int, int],
        mode: str,
    ) -> Image.Image:
        """Crop the padded canvas and resize a predicted channel to the original size."""

        pad_top, pad_left, resized_h, resized_w = crop_box
        if prediction.ndim == 2:
            cropped = prediction[
                pad_top : pad_top + resized_h, pad_left : pad_left + resized_w
            ]
            arr = cropped.clamp(0.0, 1.0).cpu().numpy()
        else:
            cropped = prediction[
                :, pad_top : pad_top + resized_h, pad_left : pad_left + resized_w
            ]
            arr = cropped.clamp(0.0, 1.0).permute(1, 2, 0).cpu().numpy()
        image = Image.fromarray((arr * 255.0).round().astype(np.uint8), mode=mode)
        return image.resize(original_size, Image.Resampling.BICUBIC)

    def _predict_batch(
        self, batch: Sequence[dict[str, Any]], generator: Any
    ) -> list[tuple[Image.Image, Image.Image, Image.Image, Image.Image]]:
        import contextlib

        import torch

        if self.pipeline is None:
            raise RuntimeError("Call setup() before predict()")

        imgs_in = torch.stack([item["img"] for item in batch]).to(self.device)
        task_ids = torch.tensor([0, 1, 2], dtype=torch.long)
        task_ids = task_ids.repeat(len(batch)).to(self.device)
        # Upstream casts the camera pose to half precision before inference;
        # a float32 pose would otherwise mix dtypes inside the CaPE embedding
        # under autocast and break xformers attention.
        pose_dtype = torch.float16 if self.device.startswith("cuda") else torch.float32
        cam_pose = torch.stack([item["pose"] for item in batch]).to(
            self.device, dtype=pose_dtype
        )

        context = (
            torch.autocast("cuda")
            if self.device.startswith("cuda")
            else contextlib.nullcontext()
        )
        with torch.no_grad(), context:
            out = self.pipeline(
                imgs_in,
                task_ids,
                num_views=1,
                cam_pose=cam_pose,
                height=_DIFFUSION_SIZE,
                width=_DIFFUSION_SIZE,
                num_inference_steps=self.num_inference_steps,
                guidance_scale=self.guidance_scale,
                num_images_per_prompt=1,
                eta=self.eta,
                generator=generator,
                output_type="pt",
            ).images

        out = out.reshape(len(batch), _NUM_DOMAINS, 3, _DIFFUSION_SIZE, _DIFFUSION_SIZE)
        results: list[tuple[Image.Image, Image.Image, Image.Image, Image.Image]] = []
        for item, prediction in zip(batch, out):
            albedo_domain, normal_domain, mro_domain = (
                prediction[0],
                prediction[1],
                prediction[2],
            )
            crop_box: tuple[int, int, int, int] = item["crop_box"]
            original_size: tuple[int, int] = item["original_size"]
            albedo = self._restore_channel(
                albedo_domain, crop_box, original_size, "RGB"
            )
            normal = self._restore_channel(
                normal_domain, crop_box, original_size, "RGB"
            )
            metallic = self._restore_channel(
                mro_domain[0], crop_box, original_size, "L"
            )
            roughness = self._restore_channel(
                mro_domain[1], crop_box, original_size, "L"
            )
            results.append((albedo, roughness, metallic, normal))
        return results

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

        prepared = [self._prepare_sample(sample) for sample in samples]

        predictions: dict[str, Prediction2D] = {}
        for start in tqdm(
            range(0, len(prepared), self.batch_size),
            desc="IDArb prediction",
            unit="batch",
        ):
            batch = prepared[start : start + self.batch_size]
            generator = (
                torch.Generator(device=self.device).manual_seed(self.seed)
                if self.seed is not None
                else None
            )
            logger.info(
                "IDArb: samples %d-%d of %d",
                start + 1,
                start + len(batch),
                len(prepared),
            )
            batch_results = self._predict_batch(batch, generator)

            for item, (albedo, roughness, metallic, normal) in zip(
                batch, batch_results
            ):
                sample = item["sample"]
                artifacts: dict[str, Any] = {}
                if normal is not None:
                    artifacts["normal"] = normal
                if sample.mask is not None:
                    artifacts["mask"] = sample.mask

                predictions[sample.sample_id] = Prediction2D(
                    albedo=albedo,
                    roughness=roughness,
                    metallic=metallic,
                    artifacts=artifacts,
                ).save(save_dir=output_dir / sample.sample_id, mark_success=True)
        return predictions
