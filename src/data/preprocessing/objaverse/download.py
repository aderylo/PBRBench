"""Download only the Objaverse GLBs named by an object split."""

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

DEFAULT_SPLIT = PROJECT_ROOT / "configs/data/splits/objaverse_pbr_64.yaml"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data/objaverse"
DEFAULT_REPO_ID = "allenai/objaverse"


def load_objects(split_path: Path, partition: str = "all") -> list[dict[str, str]]:
    split = yaml.safe_load(split_path.read_text()) or {}
    names = ("train", "val", "test") if partition == "all" else (partition,)
    objects = [item for name in names for item in split.get(name, [])]
    if not objects:
        raise ValueError(f"Object split/partition '{partition}' is empty: {split_path}")

    path_template = split.get("path_template")
    if not path_template:
        raise ValueError(f"Split has no path_template: {split_path}")

    parsed = [
        {
            "id": str(item["id"]),
            "path": str(item.get("path") or path_template.format(id=item["id"])),
        }
        for item in objects
    ]
    ids = [item["id"] for item in parsed]
    if len(ids) != len(set(ids)):
        raise ValueError(f"Object split contains duplicate ids: {split_path}")
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
    split_path: Path,
    output_dir: Path,
    repo_id: str,
    partition: str = "all",
    workers: int = 4,
) -> None:
    objects = load_objects(split_path, partition)
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
    parser.add_argument("--split", type=Path, default=DEFAULT_SPLIT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--repo-id", default=DEFAULT_REPO_ID)
    parser.add_argument(
        "--partition", choices=("all", "train", "val", "test"), default="all"
    )
    parser.add_argument("--workers", type=int, default=4)
    arguments = parser.parse_args()
    if arguments.workers < 1:
        parser.error("--workers must be at least 1")
    return arguments


if __name__ == "__main__":
    arguments = parse_args()
    download(
        arguments.split.expanduser().resolve(),
        arguments.output_dir.expanduser().resolve(),
        arguments.repo_id,
        arguments.partition,
        arguments.workers,
    )
