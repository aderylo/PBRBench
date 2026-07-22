"""Download the TexVerse-1K GLBs named by a benchmark subset.

Adapted from the neighboring LTX-2 data pipeline. Keeping the downloader next
to the dataset integration makes the source-data boundary explicit; rendered
benchmark samples are produced separately by the launchers in
``src/data/preprocessing/``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml
from huggingface_hub import hf_hub_download

DEFAULT_REPO_ID = "YiboZhang2001/TexVerse-1K"


def load_objects(subset_path: Path) -> list[dict[str, str]]:
    subset = yaml.safe_load(subset_path.read_text()) or {}
    objects = subset.get("objects", [])
    if not objects:
        raise ValueError(f"Object subset is empty: {subset_path}")

    path_template = subset.get("path_template")
    parsed = []
    for item in objects:
        object_id = str(item["id"])
        path = item.get("path")
        if path is None and path_template is not None:
            path = str(path_template).format(id=object_id)
        if path is None:
            raise ValueError(f"Object '{object_id}' has no path in subset: {subset_path}")
        parsed.append({"id": object_id, "path": str(path)})

    ids = [item["id"] for item in parsed]
    if len(ids) != len(set(ids)):
        raise ValueError(f"Object subset contains duplicate ids: {subset_path}")
    return parsed


def download(subset_path: Path, output_dir: Path, repo_id: str) -> None:
    objects = load_objects(subset_path)
    pending = [item for item in objects if not (output_dir / item["path"]).is_file()]
    if not pending:
        print(f"All {len(objects)} objects already exist in {output_dir}")
        return

    for index, item in enumerate(pending, start=1):
        print(f"[{index}/{len(pending)}] {item['id']}")
        hf_hub_download(
            repo_id=repo_id,
            filename=item["path"],
            repo_type="dataset",
            local_dir=output_dir,
        )

    print(f"Downloaded {len(pending)} objects to {output_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--subset", required=True, type=Path, help="TexVerse benchmark subset YAML"
    )
    parser.add_argument(
        "--output-dir", required=True, type=Path, help="Local TexVerse root"
    )
    parser.add_argument(
        "--repo-id", default=DEFAULT_REPO_ID, help="Hugging Face dataset repository"
    )
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    download(
        arguments.subset.expanduser().resolve(),
        arguments.output_dir.expanduser().resolve(),
        arguments.repo_id,
    )
