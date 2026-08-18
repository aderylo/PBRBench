"""Download pretrained weights for SuperMat (fine-tuned UNet and base SD 2.1 pipeline)."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / ".weights/supermat"
DEFAULT_HF_HOME = PROJECT_ROOT / ".weights/huggingface"

# Set HF_HOME before importing huggingface_hub
os.environ["HF_HOME"] = str(DEFAULT_HF_HOME)

from huggingface_hub import hf_hub_download, snapshot_download

SUPERMAT_REPO_ID = "oyiya/SuperMat"
SUPERMAT_FILENAME = "supermat.pth"
BASE_MODEL_REPO_ID = "sd2-community/stable-diffusion-2-1"


def download_supermat_weights(
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    hf_home: Path = DEFAULT_HF_HOME,
) -> Path:
    """Download SuperMat checkpoint and base SD2.1 model into local weights and HF cache."""
    output_dir.mkdir(parents=True, exist_ok=True)
    hf_home.mkdir(parents=True, exist_ok=True)
    os.environ["HF_HOME"] = str(hf_home)

    print(f"Downloading base pipeline '{BASE_MODEL_REPO_ID}' to HF cache '{hf_home}'...")
    snapshot_download(
        repo_id=BASE_MODEL_REPO_ID,
        ignore_patterns=["*.ckpt", "*.flax_model.bin", "*.msgpack"],
    )

    print(f"Downloading {SUPERMAT_FILENAME} from '{SUPERMAT_REPO_ID}' to '{output_dir}'...")
    downloaded_path = hf_hub_download(
        repo_id=SUPERMAT_REPO_ID,
        filename=SUPERMAT_FILENAME,
        local_dir=str(output_dir),
    )
    print(f"Successfully downloaded SuperMat weights to: {downloaded_path}")
    return Path(downloaded_path)


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
    download_supermat_weights(output_dir=args.output_dir, hf_home=args.hf_home)
