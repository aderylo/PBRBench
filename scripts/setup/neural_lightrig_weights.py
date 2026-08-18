"""Download pretrained weights for Neural LightRig (mld.pt, recon models, and base backbones)."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / ".weights/neural_lightrig"
DEFAULT_HF_HOME = PROJECT_ROOT / ".weights/huggingface"

# Set HF_HOME before importing huggingface_hub
os.environ["HF_HOME"] = str(DEFAULT_HF_HOME)

from huggingface_hub import snapshot_download

REPO_ID = "zxhezexin/neural-lightrig-mld-and-recon"
REVISION = "5619cfec5e623ded0701d0b05f26ad5bbf9f0401"
ALLOW_PATTERNS = ["mld.pt", "recon/*"]

BASE_SD_REPO_ID = "sd2-community/stable-diffusion-2-1"
UNCLIP_REPO_ID = "sd2-community/stable-diffusion-2-1-unclip"
CLIP_REPO_ID = "laion/CLIP-ViT-H-14-laion2B-s32B-b79K"


def download_neural_lightrig_weights(
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    hf_home: Path = DEFAULT_HF_HOME,
) -> Path:
    """Download Neural LightRig checkpoints and upstream foundation backbones."""
    output_dir.mkdir(parents=True, exist_ok=True)
    hf_home.mkdir(parents=True, exist_ok=True)
    os.environ["HF_HOME"] = str(hf_home)

    print(f"Downloading base SD model '{BASE_SD_REPO_ID}' to HF cache '{hf_home}'...")
    snapshot_download(
        repo_id=BASE_SD_REPO_ID,
        ignore_patterns=["*.ckpt", "*.flax_model.bin", "*.msgpack"],
    )

    print(f"Downloading SD2.1-unclip model '{UNCLIP_REPO_ID}' to HF cache '{hf_home}'...")
    snapshot_download(
        repo_id=UNCLIP_REPO_ID,
        ignore_patterns=["*.ckpt", "*.flax_model.bin", "*.msgpack"],
    )

    print(f"Downloading CLIP vision model '{CLIP_REPO_ID}' to HF cache '{hf_home}'...")
    snapshot_download(repo_id=CLIP_REPO_ID)

    print(f"Downloading zero123plus scheduler + feature_extractor_vae from 'sudo-ai/zero123plus-v1.2' to HF cache '{hf_home}'...")
    snapshot_download(
        repo_id="sudo-ai/zero123plus-v1.2",
        allow_patterns=["scheduler/*", "feature_extractor_vae/*"],
    )

    print(
        f"Downloading Neural LightRig weights from '{REPO_ID}' to '{output_dir}'...\n"
        "This repository is gated; authenticate first with: "
        "third_party/.venvs/neural_lightrig/bin/hf auth login"
    )
    downloaded_dir = snapshot_download(
        repo_id=REPO_ID,
        revision=REVISION,
        local_dir=str(output_dir),
        allow_patterns=ALLOW_PATTERNS,
        local_dir_use_symlinks=False,
    )
    print(f"Successfully downloaded Neural LightRig weights to: {downloaded_dir}")
    return Path(downloaded_dir)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Destination directory for model weights",
    )
    parser.add_argument(
        "--hf-home",
        type=Path,
        default=DEFAULT_HF_HOME,
        help="Destination directory for Hugging Face cache",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    download_neural_lightrig_weights(output_dir=args.output_dir, hf_home=args.hf_home)
