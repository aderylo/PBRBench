"""Download pretrained weights for IDArb."""

from __future__ import annotations

import argparse
from pathlib import Path

from huggingface_hub import snapshot_download

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / ".weights/idarb"
REPO_ID = "lizb6626/IDArb"

ALLOW_PATTERNS = (
    "unet/*",
    "vae/*",
    "text_encoder/*",
    "tokenizer/*",
    "feature_extractor/*",
    "scheduler/*",
)


def download_idarb_weights(output_dir: Path = DEFAULT_OUTPUT_DIR) -> Path:
    """Download the IDArb diffusion components from Hugging Face into output_dir."""
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {REPO_ID} into '{output_dir}'...")
    downloaded_path = snapshot_download(
        repo_id=REPO_ID,
        allow_patterns=ALLOW_PATTERNS,
        local_dir=str(output_dir),
        local_dir_use_symlinks=False,
    )
    print(f"Successfully downloaded IDArb weights to: {downloaded_path}")
    return Path(downloaded_path)


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
    download_idarb_weights(output_dir=args.output_dir)
