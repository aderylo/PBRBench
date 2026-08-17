"""Download pretrained weights for Neural LightRig (mld.pt and recon models)."""

from __future__ import annotations

import argparse
from pathlib import Path

from huggingface_hub import snapshot_download

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / ".weights/neural_lightrig"
REPO_ID = "zxhezexin/neural-lightrig-mld-and-recon"
REVISION = "5619cfec5e623ded0701d0b05f26ad5bbf9f0401"
ALLOW_PATTERNS = ["mld.pt", "recon/*"]


def download_neural_lightrig_weights(output_dir: Path = DEFAULT_OUTPUT_DIR) -> Path:
    """Download the MLD and reconstruction checkpoints from the gated Hugging Face repository."""
    output_dir.mkdir(parents=True, exist_ok=True)
    print(
        f"Downloading Neural LightRig weights from '{REPO_ID}' to '{output_dir}'...\n"
        "This repository is gated; authenticate first with: "
        "third_party/.venvs/neural_lightrig/bin/hf auth login"
    )
    downloaded_dir = snapshot_download(
        repo_id=REPO_ID,
        revision=REVISION,
        local_dir=str(output_dir),
        allow_patterns=ALLOW_PATTERNS,
        local_dir_use_symlinks=False,
    )
    print(f"Successfully downloaded Neural LightRig weights to: {downloaded_dir}")
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
    download_neural_lightrig_weights(output_dir=args.output_dir)
