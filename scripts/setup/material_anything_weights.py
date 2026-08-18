"""Download pretrained weights for MaterialAnything (estimator, UV refiner, and RMBG background remover)."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from huggingface_hub import snapshot_download

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / ".weights/material_anything"
DEFAULT_HF_HOME = PROJECT_ROOT / ".weights/huggingface"

MODELS = {
    "material_estimator": "xanderhuang/material_estimator",
    "material_refiner": "xanderhuang/material_refiner",
}
RMBG_REPO_ID = "briaai/RMBG-2.0"


def download_material_anything_weights(
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    hf_home: Path = DEFAULT_HF_HOME,
) -> Path:
    """Download the material estimator, UV refiner, and RMBG checkpoints into output_dir and HF cache."""
    output_dir.mkdir(parents=True, exist_ok=True)
    hf_home.mkdir(parents=True, exist_ok=True)
    os.environ["HF_HOME"] = str(hf_home)

    print(f"Downloading RMBG background remover from '{RMBG_REPO_ID}' to HF cache '{hf_home}'...")
    try:
        snapshot_download(repo_id=RMBG_REPO_ID)
    except Exception as exc:
        print(f"Note: RMBG download skipped/failed ({exc}).")

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
    parser.add_argument(
        "--hf-home",
        type=Path,
        default=DEFAULT_HF_HOME,
        help="Destination directory for Hugging Face cache",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    download_material_anything_weights(output_dir=args.output_dir, hf_home=args.hf_home)
