"""Download pretrained weights for Hunyuan3D-2.1 texturing model (Hunyuan3D-Paint 2.1)."""

from __future__ import annotations

import argparse
import logging
import urllib.request
from pathlib import Path

from huggingface_hub import snapshot_download

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / ".weights/hunyuan3d"
DEFAULT_REPO_ID = "tencent/Hunyuan3D-2.1"
DINO_REPO_ID = "facebook/dinov2-giant"
REALESRGAN_URL = "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth"

PAINTPBR_PATTERNS = [
    "hunyuan3d-paintpbr-v2-1/*",
]

logger = logging.getLogger(__name__)


def download_realesrgan_weight(ckpt_dir: Path) -> Path:
    """Download RealESRGAN_x4plus.pth if not already present."""
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    target_path = ckpt_dir / "RealESRGAN_x4plus.pth"
    if target_path.exists() and target_path.stat().st_size > 0:
        print(f"RealESRGAN weight already exists at: {target_path}")
        return target_path

    print(f"Downloading RealESRGAN weight from {REALESRGAN_URL} to {target_path}...")
    urllib.request.urlretrieve(REALESRGAN_URL, target_path)
    print(f"Successfully downloaded RealESRGAN weight to: {target_path}")
    return target_path


def download_hunyuan3d_weights(
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    repo_id: str = DEFAULT_REPO_ID,
    all_weights: bool = False,
    download_dino: bool = True,
    download_realesrgan: bool = True,
) -> Path:
    """Download Hunyuan3D-2.1 checkpoints from Hugging Face into output_dir."""
    output_dir.mkdir(parents=True, exist_ok=True)
    allow_patterns = None if all_weights else PAINTPBR_PATTERNS

    print(
        f"Downloading Hunyuan3D-2.1 weights ({'all' if all_weights else 'hy3dpaint PBR pipeline'}) "
        f"from '{repo_id}' to '{output_dir}'..."
    )
    downloaded_dir = snapshot_download(
        repo_id=repo_id,
        local_dir=str(output_dir),
        allow_patterns=allow_patterns,
        local_dir_use_symlinks=False,
    )
    print(f"Successfully downloaded Hunyuan3D weights to: {downloaded_dir}")

    if download_dino:
        dino_dir = output_dir / "dinov2-giant"
        print(
            f"Downloading DINOv2-Giant weights from '{DINO_REPO_ID}' to '{dino_dir}'..."
        )
        snapshot_download(
            repo_id=DINO_REPO_ID,
            local_dir=str(dino_dir),
            local_dir_use_symlinks=False,
        )
        print(f"Successfully downloaded DINOv2-Giant weights to: {dino_dir}")

    if download_realesrgan:
        download_realesrgan_weight(output_dir / "ckpt")

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
        "--repo-id",
        type=str,
        default=DEFAULT_REPO_ID,
        help="Hugging Face repository ID for Hunyuan3D-2.1",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Download all weights (including full 3D shape generation), not just texturing pipeline",
    )
    parser.add_argument(
        "--no-dino",
        action="store_true",
        help="Skip downloading DINOv2-Giant weights",
    )
    parser.add_argument(
        "--no-realesrgan",
        action="store_true",
        help="Skip downloading RealESRGAN weights",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    download_hunyuan3d_weights(
        output_dir=args.output_dir,
        repo_id=args.repo_id,
        all_weights=args.all,
        download_dino=not args.no_dino,
        download_realesrgan=not args.no_realesrgan,
    )
