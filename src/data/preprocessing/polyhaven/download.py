"""Download the Poly Haven assets named by a benchmark subset as GLBs.

The Poly Haven API supplies a glTF manifest with a model file and its texture
and binary dependencies. This downloader verifies the complete source bundle,
then embeds it into one self-contained GLB at the canonical asset path.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import tempfile
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

import requests
import rootutils
import yaml

PROJECT_ROOT = rootutils.setup_root(__file__, indicator=".project_root", pythonpath=True)

from src.data.preprocessing._assets import canonical_asset_path, gltf_to_glb  # noqa: E402

DEFAULT_SUBSET = PROJECT_ROOT / "configs/data/subsets/polyhaven_pbr_64.yaml"
DEFAULT_ASSETS_ROOT = PROJECT_ROOT / "data/assets"
DEFAULT_RESOLUTION = "1k"
SOURCE_NAME = "polyhaven"


@dataclass(frozen=True)
class FileSpec:
    """One file in a Poly Haven glTF asset bundle."""

    relative_path: Path
    url: str
    md5: str


@dataclass(frozen=True)
class AssetSpec:
    """The complete model bundle needed to import a Poly Haven asset."""

    object_id: str
    files: tuple[FileSpec, ...]


def load_subset_ids(subset_path: Path) -> list[str]:
    """Load unique asset IDs from a benchmark subset YAML."""
    subset = yaml.safe_load(subset_path.read_text()) or {}
    objects = subset.get("objects", [])
    if not objects:
        raise ValueError(f"Object subset is empty: {subset_path}")

    object_ids = [str(item["id"]) for item in objects]
    if len(object_ids) != len(set(object_ids)):
        raise ValueError(f"Object subset contains duplicate ids: {subset_path}")
    return object_ids


def safe_relative_path(value: str) -> Path:
    """Convert an API-provided POSIX path to a traversal-safe local path."""
    path = PurePosixPath(value)
    if not value or path.is_absolute() or ".." in path.parts or "." in path.parts:
        raise ValueError(f"Poly Haven manifest contains unsafe relative path: {value!r}")
    return Path(*path.parts)


def file_spec(relative_path: str, payload: Any) -> FileSpec:
    """Validate one file record from the Poly Haven API."""
    if not isinstance(payload, dict) or not all(
        isinstance(payload.get(field), str) and payload[field]
        for field in ("url", "md5")
    ):
        raise ValueError(f"Malformed Poly Haven file record for {relative_path}")
    return FileSpec(
        relative_path=safe_relative_path(relative_path),
        url=payload["url"],
        md5=payload["md5"].lower(),
    )


def fetch_asset_spec(object_id: str, resolution: str, timeout: float) -> AssetSpec:
    """Fetch one asset's glTF bundle manifest from the public Poly Haven API."""
    response = requests.get(
        f"https://api.polyhaven.com/files/{object_id}", timeout=timeout
    )
    response.raise_for_status()
    try:
        model = response.json()["gltf"][resolution]["gltf"]
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(
            f"Poly Haven asset {object_id!r} has no glTF at resolution {resolution!r}"
        ) from error

    primary = file_spec("asset.gltf", model)
    includes = model.get("include", {}) if isinstance(model, dict) else {}
    if not isinstance(includes, dict):
        raise ValueError(f"Malformed glTF dependency list for {object_id!r}")
    dependencies = [file_spec(path, item) for path, item in includes.items()]
    return AssetSpec(object_id=object_id, files=tuple([primary, *dependencies]))


def md5_matches(path: Path, expected_md5: str) -> bool:
    """Check a file's API-provided checksum incrementally."""
    digest = hashlib.md5()  # noqa: S324 - Poly Haven publishes MD5 checksums.
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest() == expected_md5.lower()


def download_file(spec: FileSpec, destination: Path, timeout: float) -> str:
    """Download, verify, and atomically install one model-bundle file."""
    if destination.is_file() and md5_matches(destination, spec.md5):
        return "cached"

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{uuid.uuid4().hex}.part")
    digest = hashlib.md5()  # noqa: S324 - Poly Haven publishes MD5 checksums.
    try:
        with requests.get(spec.url, stream=True, timeout=timeout) as response:
            response.raise_for_status()
            with temporary.open("wb") as file:
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        file.write(chunk)
                        digest.update(chunk)
        if digest.hexdigest() != spec.md5.lower():
            raise ValueError(f"MD5 mismatch for {destination}")
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    return "downloaded"


def download_asset(spec: AssetSpec, assets_root: Path, timeout: float) -> str:
    """Verify a Poly Haven glTF bundle and install it as one canonical GLB."""
    destination = canonical_asset_path(assets_root, SOURCE_NAME, spec.object_id)
    if destination.is_file():
        return "cached"

    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        dir=destination.parent, prefix=f".{destination.stem}."
    ) as temporary_directory:
        bundle_root = Path(temporary_directory)
        for file in spec.files:
            download_file(file, bundle_root / file.relative_path, timeout)
        temporary_glb = bundle_root / destination.name
        gltf_to_glb(bundle_root / "asset.gltf", temporary_glb)
        os.replace(temporary_glb, destination)
    return "downloaded"


def download(
    subset_path: Path,
    assets_root: Path,
    resolution: str = DEFAULT_RESOLUTION,
    workers: int = 4,
    timeout: float = 60.0,
    dry_run: bool = False,
) -> None:
    """Download the selected Poly Haven assets as self-contained GLBs."""
    object_ids = load_subset_ids(subset_path)
    specs = [fetch_asset_spec(object_id, resolution, timeout) for object_id in object_ids]
    if dry_run:
        for spec in specs:
            print(
                f"{spec.object_id} -> "
                f"{canonical_asset_path(assets_root, SOURCE_NAME, spec.object_id)} "
                f"({len(spec.files)} source files)"
            )
        return

    assets_root.mkdir(parents=True, exist_ok=True)
    failures: list[str] = []
    results: dict[str, str] = {}
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(download_asset, spec, assets_root, timeout): spec.object_id
            for spec in specs
        }
        for index, future in enumerate(as_completed(futures), start=1):
            object_id = futures[future]
            try:
                results[object_id] = future.result()
                print(f"[{index}/{len(specs)}] {object_id}: {results[object_id]}")
            except (OSError, requests.RequestException, ValueError) as error:
                failures.append(f"{object_id}: {error}")

    if failures:
        raise RuntimeError("Failed Poly Haven assets:\n- " + "\n- ".join(failures))
    print(
        f"Downloaded {sum(result == 'downloaded' for result in results.values())} "
        f"Poly Haven assets to {assets_root / SOURCE_NAME} "
        f"({sum(result == 'cached' for result in results.values())} already present)"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subset", type=Path, default=DEFAULT_SUBSET)
    parser.add_argument(
        "--assets-root",
        type=Path,
        default=DEFAULT_ASSETS_ROOT,
        help="Canonical source-asset root; files go to data/assets/polyhaven/<id>.glb",
    )
    parser.add_argument("--resolution", default=DEFAULT_RESOLUTION)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument("--dry-run", action="store_true")
    arguments = parser.parse_args()
    if arguments.workers < 1:
        parser.error("--workers must be at least 1")
    if arguments.timeout <= 0:
        parser.error("--timeout must be positive")
    return arguments


if __name__ == "__main__":
    arguments = parse_args()
    download(
        subset_path=arguments.subset.expanduser().resolve(),
        assets_root=arguments.assets_root.expanduser().resolve(),
        resolution=arguments.resolution,
        workers=arguments.workers,
        timeout=arguments.timeout,
        dry_run=arguments.dry_run,
    )
