"""Download pretrained weights for MaterialAnything (estimator and UV refiner)."""

from __future__ import annotations

import argparse
from pathlib import Path

from huggingface_hub import snapshot_download

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / ".weights/material_anything"
MODELS = {
    "material_estimator": "xanderhuang/material_estimator",
    "material_refiner": "xanderhuang/material_refiner",
}


def download_material_anything_weights(output_dir: Path = DEFAULT_OUTPUT_DIR) -> Path:
    """Download the material estimator and UV refiner checkpoints into output_dir."""
    output_dir.mkdir(parents=True, exist_ok=True)
    for folder_name, repo_id in MODELS.items():
        local_dir = output_dir / folder_name
        print(f"Downloading '{repo_id}' to '{local_dir}'...")
        snapshot_download(
            repo_id=repo_id,
            local_dir=str(local_dir),
            local_dir_use_symlinks=False,
        )
        print(f"Successfully downloaded {folder_name} weights to: {local_dir}")
    return output_dir


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
    download_material_anything_weights(output_dir=args.output_dir)
