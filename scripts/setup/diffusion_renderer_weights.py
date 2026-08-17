"""Download pretrained weights for DiffusionRenderer."""

from __future__ import annotations

import argparse
from pathlib import Path

from huggingface_hub import snapshot_download

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / ".weights/diffusion_renderer"
REPO_ID = "nexuslrf/diffusion_renderer-inverse-svd"


def download_diffusion_renderer_weights(output_dir: Path = DEFAULT_OUTPUT_DIR) -> Path:
    """Download the DiffusionRenderer inverse-rendering checkpoint from Hugging Face."""
    output_dir.mkdir(parents=True, exist_ok=True)
    print(
        f"Downloading DiffusionRenderer weights from '{REPO_ID}' to '{output_dir}'..."
    )
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
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    download_diffusion_renderer_weights(output_dir=args.output_dir)
