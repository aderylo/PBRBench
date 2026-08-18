"""Download pretrained weights for TRELLIS 2 and upstream backbones (BiRefNet, DINOv3)."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from huggingface_hub import snapshot_download

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / ".weights/trellis2"
DEFAULT_HF_HOME = PROJECT_ROOT / ".weights/huggingface"

REPO_ID = "microsoft/TRELLIS.2-4B"
BIREFNET_REPO_ID = "ZhengPeng7/BiRefNet"
DINOV3_REPO_ID = "facebook/dinov3-vitl16-pretrain-lvd1689m"


def download_trellis2_weights(
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    hf_home: Path = DEFAULT_HF_HOME,
) -> Path:
    """Download TRELLIS.2-4B weights and its upstream backbone dependencies."""
    output_dir.mkdir(parents=True, exist_ok=True)
    hf_home.mkdir(parents=True, exist_ok=True)
    os.environ["HF_HOME"] = str(hf_home)

    print(f"Downloading BiRefNet backbone from '{BIREFNET_REPO_ID}' to HF cache '{hf_home}'...")
    snapshot_download(repo_id=BIREFNET_REPO_ID)

    try:
        print(f"Downloading DINOv3 backbone from '{DINOV3_REPO_ID}' to HF cache '{hf_home}'...")
        snapshot_download(repo_id=DINOV3_REPO_ID)
    except Exception as exc:
        print(f"Note: DINOv3 download skipped/failed ({exc}).")

    print(f"Downloading TRELLIS 2 weights from '{REPO_ID}' to '{output_dir}'...")
    downloaded_dir = snapshot_download(
        repo_id=REPO_ID,
        local_dir=str(output_dir),
        local_dir_use_symlinks=False,
    )
    print(f"Successfully downloaded TRELLIS 2 weights to: {downloaded_dir}")
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
    download_trellis2_weights(output_dir=args.output_dir, hf_home=args.hf_home)
