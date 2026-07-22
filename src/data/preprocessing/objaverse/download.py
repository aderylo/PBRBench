"""Download only the Objaverse GLBs named by a benchmark subset."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import rootutils
import yaml
from huggingface_hub import hf_hub_download

PROJECT_ROOT = rootutils.setup_root(
    __file__, indicator=".project_root", pythonpath=True
)

DEFAULT_SUBSET = PROJECT_ROOT / "configs/data/subsets/objaverse_pbr_64.yaml"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data/objaverse"
DEFAULT_REPO_ID = "allenai/objaverse"
DEFAULT_PATH_TEMPLATE = "glbs/000-001/{id}.glb"


def load_objects(subset_path: Path, path_template: str) -> list[dict[str, str]]:
    subset = yaml.safe_load(subset_path.read_text()) or {}
    objects = subset.get("objects", [])
    if not objects:
        raise ValueError(f"Object subset is empty: {subset_path}")

    parsed = [
        {
            "id": str(item["id"]),
            "path": str(item.get("path") or path_template.format(id=item["id"])),
        }
        for item in objects
    ]
    ids = [item["id"] for item in parsed]
    if len(ids) != len(set(ids)):
        raise ValueError(f"Object subset contains duplicate ids: {subset_path}")
    return parsed


def download_one(item: dict[str, str], output_dir: Path, repo_id: str) -> str:
    hf_hub_download(
        repo_id=repo_id,
        filename=item["path"],
        repo_type="dataset",
        local_dir=output_dir,
    )
    return item["id"]


def download(
    subset_path: Path,
    output_dir: Path,
    repo_id: str,
    path_template: str,
    workers: int = 4,
) -> None:
    objects = load_objects(subset_path, path_template)
    pending = [item for item in objects if not (output_dir / item["path"]).is_file()]
    if not pending:
        print(f"All {len(objects)} requested objects already exist in {output_dir}")
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(download_one, item, output_dir, repo_id): item
            for item in pending
        }
        for index, future in enumerate(as_completed(futures), start=1):
            item = futures[future]
            future.result()
            print(f"[{index}/{len(pending)}] {item['id']}")

    print(f"Downloaded {len(pending)} objects to {output_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subset", type=Path, default=DEFAULT_SUBSET)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--repo-id", default=DEFAULT_REPO_ID)
    parser.add_argument("--path-template", default=DEFAULT_PATH_TEMPLATE)
    parser.add_argument("--workers", type=int, default=4)
    arguments = parser.parse_args()
    if arguments.workers < 1:
        parser.error("--workers must be at least 1")
    return arguments


if __name__ == "__main__":
    arguments = parse_args()
    download(
        arguments.subset.expanduser().resolve(),
        arguments.output_dir.expanduser().resolve(),
        arguments.repo_id,
        arguments.path_template,
        arguments.workers,
    )
