"""Neural LightRig screen-space material estimator.

Neural LightRig (He et al., 2024) estimates surface normals and PBR materials
(albedo, roughness, metallic) from a single monocular image by leveraging a
two-stage architecture:

1. Stage I (Multi-Light Diffusion / Relighting): Generates L=9 multi-light images
   arranged in a 3x3 grid layout using a fine-tuned Stable Diffusion 2.1-unclip
   backbone with hybrid conditioning (concat latents + reference self-attention).
2. Stage II (Large G-Buffer Model): Takes the input image concatenated with the
   9 generated relit reference images and regresses surface normals and PBR material
   maps using a UNet backbone.
"""

from __future__ import annotations

import logging
import os
import random
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from unittest.mock import patch

from PIL import Image
from tqdm.auto import tqdm

from src.data.pbr_estimation_dataset_2d import PBREstimationSample2D
from src.methods_2d.base import BaseMaterialEstimator2D, Prediction2D


class NeuralLightRig2D(BaseMaterialEstimator2D):
    def __init__(
        self,
        *,
        base_model: str = "sd2-community/stable-diffusion-2-1",
        unclip_model: str = "sd2-community/stable-diffusion-2-1-unclip",
        checkpoint_repo: str = "zxhezexin/neural-lightrig-mld-and-recon",
        checkpoint_revision: str | None = "5619cfec5e623ded0701d0b05f26ad5bbf9f0401",
        checkpoint_dir: str | Path | None = None,
        input_resolution: int = 512,
        guidance_scale: float = 2.0,
        guidance_rescale: float = 0.7,
        inference_steps: int = 75,
        seed: int = 511,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.base_model = base_model
        self.unclip_model = unclip_model
        self.checkpoint_repo = checkpoint_repo
        self.checkpoint_revision = checkpoint_revision
        self.checkpoint_dir = (
            self.repo_root / "ckpt"
            if checkpoint_dir is None
            else self.resolve_path(checkpoint_dir)
        )
        self.input_resolution = input_resolution
        self.guidance_scale = guidance_scale
        self.guidance_rescale = guidance_rescale
        self.inference_steps = inference_steps
        self.seed = seed
        self.relighting_model = None
        self.recon_model = None

    def setup(self) -> None:
        super().setup()
        sys.path.insert(0, str(self.repo_root))
        os.environ["HF_HUB_DISABLE_XET"] = "1"
        os.environ["TORCH_COMPILE_DISABLE"] = "1"

        # Quiet verbose HTTP loggers
        logging.getLogger("httpx").setLevel(logging.WARNING)

        import numpy as np
        import torch
        from huggingface_hub import snapshot_download
        from omegaconf import OmegaConf

        from mld.model import MVDiffusion
        from recon.model import PBRReconConfig, PBRUNetModelForReconstruction
        from transformers import CLIPVisionModelWithProjection

        mld_checkpoint = self.checkpoint_dir / "mld.pt"
        recon_checkpoint = self.checkpoint_dir / "recon"

        # Download checkpoint weights only if not cached locally
        if not mld_checkpoint.is_file() or not recon_checkpoint.is_dir():
            snapshot_download(
                repo_id=self.checkpoint_repo,
                revision=self.checkpoint_revision,
                local_dir=self.checkpoint_dir,
                allow_patterns=["mld.pt", "recon/*"],
            )

        # Force HF offline mode once files are local so network isn't pinged on GPU nodes
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        os.environ["DIFFUSERS_OFFLINE"] = "1"

        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)

        # Stage I: Multi-Light Relighting Diffusion Model
        model_config = OmegaConf.load(self.repo_root / "mld/configs/infer.yaml")
        model_config.model.params.sd_21_pretrain = self.base_model
        load_vision_model = CLIPVisionModelWithProjection.from_pretrained

        def load_community_unclip(repo_id, *args, **kwargs):
            if repo_id == "stabilityai/stable-diffusion-2-1-unclip":
                repo_id = self.unclip_model
            return load_vision_model(repo_id, *args, **kwargs)

        with patch.object(
            CLIPVisionModelWithProjection,
            "from_pretrained",
            side_effect=load_community_unclip,
        ):
            self.relighting_model = MVDiffusion.load_from_checkpoint(
                mld_checkpoint,
                strict=True,
                map_location="cpu",
                **model_config.model.params,
            )
        self.relighting_model.eval()

        # Stage II: Large G-Buffer Surface Normal & PBR Material Regression UNet
        recon_config = PBRReconConfig.from_pretrained(
            str(recon_checkpoint), local_files_only=True
        )
        self.recon_model = PBRUNetModelForReconstruction.from_pretrained(
            str(recon_checkpoint), config=recon_config, local_files_only=True
        ).float()
        self.recon_model.eval()

    def teardown(self) -> None:
        import torch

        self.relighting_model = None
        self.recon_model = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def predict(
        self,
        samples: Sequence[PBREstimationSample2D],
        output_dir: Path,
    ) -> Mapping[str, Prediction2D]:
        if self.relighting_model is None or self.recon_model is None:
            raise RuntimeError("Call setup() before predict()")

        if not samples:
            return {}

        import torch
        from utils.vis import replace_bg_preserving_alpha

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # ----------------------------------------------------------------------
        # Stage I: Multi-Light Relighting Diffusion across all batch samples
        # ----------------------------------------------------------------------
        self.relighting_model.to(device)
        self.relighting_model.pipeline.to(device)

        ref_images: dict[str, Image.Image] = {}
        for sample in tqdm(
            samples,
            desc=f"Neural LightRig [Stage 1/2: Relighting ({len(samples)} samples)]",
            unit="sample",
        ):
            img_in_rgba = Image.open(sample.rgb).resize(
                (self.input_resolution, self.input_resolution)
            )
            img_in_rgba_white = replace_bg_preserving_alpha(img_in_rgba, 255)

            with torch.no_grad():
                img_ref: Image.Image = self.relighting_model.pipeline(
                    image=img_in_rgba_white.convert("RGB"),
                    guidance_scale=self.guidance_scale,
                    guidance_rescale=self.guidance_rescale,
                    num_inference_steps=self.inference_steps,
                ).images[0]

            ref_images[sample.sample_id] = img_ref

        # Offload Stage I model to CPU to free VRAM for Stage II
        self.relighting_model.to("cpu")
        self.relighting_model.pipeline.to("cpu")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # ----------------------------------------------------------------------
        # Stage II: G-Buffer Reconstruction (Normal & PBR Materials)
        # ----------------------------------------------------------------------
        self.recon_model.to(device)

        outputs = {}
        for sample in tqdm(
            samples,
            desc=f"Neural LightRig [Stage 2/2: Reconstruction ({len(samples)} samples)]",
            unit="sample",
        ):
            sample_dir = output_dir / sample.sample_id
            sample_dir.mkdir(parents=True, exist_ok=True)

            img_in_rgba = Image.open(sample.rgb).resize(
                (self.input_resolution, self.input_resolution)
            )
            img_in_rgba_black = replace_bg_preserving_alpha(img_in_rgba, 0)
            img_ref = ref_images[sample.sample_id]

            with torch.no_grad():
                img_rm, img_albedo, img_normal = self.recon_model.predict(
                    input_image=img_in_rgba_black,
                    ref_image=img_ref,
                )

            # Channel extraction
            albedo_rgb = img_albedo.convert("RGB")
            roughness = img_rm.getchannel("G")
            metallic = img_rm.getchannel("B")
            normal_rgb = img_normal.convert("RGB")
            mask_a = img_in_rgba.getchannel("A")
            ref_rgb = img_ref.convert("RGB")

            # Composite 5-panel output (Input, Ref, RM, Albedo, Normal)
            img_out = Image.new(
                "RGBA", (self.input_resolution * 5, self.input_resolution)
            )
            img_out.paste(img_in_rgba, (0, 0))
            img_out.paste(
                img_ref.resize((self.input_resolution, self.input_resolution)),
                (self.input_resolution, 0),
            )
            img_out.paste(img_rm, (self.input_resolution * 2, 0))
            img_out.paste(img_albedo, (self.input_resolution * 3, 0))
            img_out.paste(img_normal, (self.input_resolution * 4, 0))

            outputs[sample.sample_id] = self.save_prediction(
                sample_dir=sample_dir,
                albedo=albedo_rgb,
                roughness=roughness,
                metallic=metallic,
                normal=normal_rgb,
                artifacts={
                    "mask": mask_a,
                    "ref": ref_rgb,
                    "combined": img_out,
                },
            )

        # Offload Stage II model back to CPU after batch completion
        self.recon_model.to("cpu")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return outputs
