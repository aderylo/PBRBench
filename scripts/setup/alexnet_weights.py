"""Pre-download AlexNet weights for offline LPIPS evaluation.

This script MUST be run on an internet-connected node (e.g., login node) before running
evaluation scripts on offline SLURM compute nodes.

When indirect evaluation (Blender re-rendering / LPIPS metric computation) executes,
LPIPS requires torchvision's pretrained AlexNet backbone. This script downloads the
required model weights directly into the project caches (.weights/torch and data/checkpoints/torch).
"""

from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TORCH_HOME = PROJECT_ROOT / ".weights/torch"
CONFIG_CACHE_DIR = PROJECT_ROOT / "data/checkpoints/torch"
ALEXNET_URL = "https://download.pytorch.org/models/alexnet-owt-7be5be79.pth"


def download_alexnet_weights(torch_home: Path = DEFAULT_TORCH_HOME) -> Path:
    """Download AlexNet weights into the project torch hub cache and evaluation cache."""
    checkpoints_dir = torch_home / "hub/checkpoints"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    destination = checkpoints_dir / "alexnet-owt-7be5be79.pth"

    if not destination.is_file() or destination.stat().st_size == 0:
        print(f"Downloading AlexNet weights from '{ALEXNET_URL}' to '{destination}'...")
        os.environ["TORCH_HOME"] = str(torch_home)
        torch.hub.download_url_to_file(ALEXNET_URL, str(destination), progress=True)
        print(f"Successfully downloaded AlexNet weights to: {destination}")
    else:
        print(f"AlexNet weights already cached at: {destination}")

    # Also link/copy to configs model_cache_dir (data/checkpoints/torch/checkpoints)
    config_checkpoints_dir = CONFIG_CACHE_DIR / "checkpoints"
    config_checkpoints_dir.mkdir(parents=True, exist_ok=True)
    config_dest = config_checkpoints_dir / "alexnet-owt-7be5be79.pth"
    if not config_dest.is_file() or config_dest.stat().st_size == 0:
        print(f"Mirroring AlexNet weights to config cache: {config_dest}")
        shutil.copy2(destination, config_dest)

    return destination


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--torch-home",
        type=Path,
        default=DEFAULT_TORCH_HOME,
        help="Destination directory for PyTorch hub cache",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    download_alexnet_weights(torch_home=args.torch_home)
