"""Download pretrained weights for DiffusionRenderer (inverse-SVD and base SVD image encoder)."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / ".weights/diffusion_renderer"
DEFAULT_HF_HOME = PROJECT_ROOT / ".weights/huggingface"

# Set HF_HOME before importing huggingface_hub
os.environ["HF_HOME"] = str(DEFAULT_HF_HOME)

from huggingface_hub import snapshot_download

REPO_ID = "nexuslrf/diffusion_renderer-inverse-svd"
BASE_SVD_REPO_ID = "stabilityai/stable-video-diffusion-img2vid"
BASE_SVD_ALLOW_PATTERNS = ["image_encoder/*", "image_processor/*"]


def download_diffusion_renderer_weights(
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    hf_home: Path = DEFAULT_HF_HOME,
) -> Path:
    """Download DiffusionRenderer inverse-rendering checkpoint and base SVD encoder into HF cache."""
    output_dir.mkdir(parents=True, exist_ok=True)
    hf_home.mkdir(parents=True, exist_ok=True)
    os.environ["HF_HOME"] = str(hf_home)

    print(f"Downloading base SVD encoder components from '{BASE_SVD_REPO_ID}' to HF cache '{hf_home}'...")
    snapshot_download(
        repo_id=BASE_SVD_REPO_ID,
        allow_patterns=BASE_SVD_ALLOW_PATTERNS,
    )

    print(f"Downloading DiffusionRenderer weights from '{REPO_ID}' to '{output_dir}'...")
    downloaded_dir = snapshot_download(
        repo_id=REPO_ID,
        local_dir=str(output_dir),
        local_dir_use_symlinks=False,
    )
    print(f"Successfully downloaded DiffusionRenderer weights to: {downloaded_dir}")
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
    download_diffusion_renderer_weights(output_dir=args.output_dir, hf_home=args.hf_home)
