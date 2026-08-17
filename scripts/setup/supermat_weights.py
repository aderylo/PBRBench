"""Download pretrained weights for SuperMat."""

from __future__ import annotations

import argparse
from pathlib import Path

from huggingface_hub import hf_hub_download

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / ".weights/supermat"
REPO_ID = "oyiya/SuperMat"
FILENAME = "supermat.pth"


def download_supermat_weights(output_dir: Path = DEFAULT_OUTPUT_DIR) -> Path:
    """Download the SuperMat checkpoint from Hugging Face into output_dir."""
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {FILENAME} from '{REPO_ID}' to '{output_dir}'...")
    downloaded_path = hf_hub_download(
        repo_id=REPO_ID,
        filename=FILENAME,
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
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    download_supermat_weights(output_dir=args.output_dir)
