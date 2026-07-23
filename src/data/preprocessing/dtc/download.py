"""Download the DTC GLBs selected by a benchmark subset.

The DTC download-URL manifest is granted after accepting the dataset licence and
contains short-lived URLs.  It is intentionally supplied at run time rather
than committed to this repository.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

import requests
import rootutils
import yaml

PROJECT_ROOT = rootutils.setup_root(__file__, indicator=".project_root", pythonpath=True)

from src.data.preprocessing._assets import canonical_asset_path  # noqa: E402

DEFAULT_SUBSET = PROJECT_ROOT / "configs/data/subsets/dtc_pbr_64.yaml"
DEFAULT_ASSETS_ROOT = PROJECT_ROOT / "data/assets"
SOURCE_NAME = "dtc"
DEFAULT_RELEASE = "DTC"
DEFAULT_ASSET_KEY = "3d-asset_glb"


@dataclass(frozen=True)
class DownloadSpec:
    """The temporary URL and integrity data needed for one source asset."""

    object_id: str
    url: str
    sha1sum: str


def load_subset_ids(subset_path: Path) -> list[str]:
    """Load unique object IDs from a benchmark subset YAML."""
    subset = yaml.safe_load(subset_path.read_text()) or {}
    objects = subset.get("objects", [])
    if not objects:
        raise ValueError(f"Object subset is empty: {subset_path}")

    object_ids = [str(item["id"]) for item in objects]
    if len(object_ids) != len(set(object_ids)):
        raise ValueError(f"Object subset contains duplicate ids: {subset_path}")
    return object_ids


def load_download_specs(
    download_urls_path: Path,
    object_ids: list[str],
    release: str,
    asset_key: str,
) -> list[DownloadSpec]:
    """Resolve subset IDs against a DTC URL manifest without persisting URLs."""
    try:
        payload = json.loads(download_urls_path.read_text())
        available_objects = payload["releases"][release]["objects"]
    except (json.JSONDecodeError, KeyError, TypeError) as error:
        raise ValueError(
            "Expected a DTC download manifest with "
            f"'releases.{release}.objects': {download_urls_path}"
        ) from error
    if not isinstance(available_objects, dict):
        raise ValueError(
            "Expected DTC manifest objects to be a mapping: "
            f"{download_urls_path}"
        )

    specs = []
    missing = []
    malformed = []
    for object_id in object_ids:
        asset = available_objects.get(object_id, {}).get(asset_key)
        if asset is None:
            missing.append(object_id)
            continue
        if not isinstance(asset, dict) or not all(
            isinstance(asset.get(field), str) and asset[field]
            for field in ("download_url", "sha1sum")
        ):
            malformed.append(object_id)
            continue
        specs.append(
            DownloadSpec(
                object_id=object_id,
                url=asset["download_url"],
                sha1sum=asset["sha1sum"].lower(),
            )
        )

    if missing or malformed:
        details = []
        if missing:
            details.append(f"missing {asset_key}: {', '.join(missing)}")
        if malformed:
            details.append(f"malformed {asset_key}: {', '.join(malformed)}")
        raise ValueError("; ".join(details))
    return specs


def sha1_matches(path: Path, expected_sha1: str) -> bool:
    """Check a file's SHA-1 incrementally, without loading it into memory."""
    digest = hashlib.sha1()  # noqa: S324 - DTC publishes SHA-1 checksums.
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest() == expected_sha1.lower()


def download_one(spec: DownloadSpec, assets_root: Path, timeout: float) -> str:
    """Download, verify, and atomically install one DTC GLB."""
    destination = canonical_asset_path(assets_root, SOURCE_NAME, spec.object_id)
    if destination.is_file() and sha1_matches(destination, spec.sha1sum):
        return "cached"

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{uuid.uuid4().hex}.part")
    digest = hashlib.sha1()  # noqa: S324 - DTC publishes SHA-1 checksums.
    try:
        with requests.get(spec.url, stream=True, timeout=timeout) as response:
            response.raise_for_status()
            with temporary.open("wb") as file:
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        file.write(chunk)
                        digest.update(chunk)
        if digest.hexdigest() != spec.sha1sum.lower():
            raise ValueError(
                f"SHA-1 mismatch for {spec.object_id}; refresh the DTC URL manifest "
                "and retry"
            )
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    return "downloaded"


def download(
    subset_path: Path,
    download_urls_path: Path,
    assets_root: Path,
    release: str = DEFAULT_RELEASE,
    asset_key: str = DEFAULT_ASSET_KEY,
    workers: int = 4,
    timeout: float = 60.0,
    dry_run: bool = False,
) -> None:
    """Download every selected DTC asset into the configured local layout."""
    object_ids = load_subset_ids(subset_path)
    specs = load_download_specs(download_urls_path, object_ids, release, asset_key)
    if dry_run:
        for spec in specs:
            print(
                f"{spec.object_id} -> "
                f"{canonical_asset_path(assets_root, SOURCE_NAME, spec.object_id)}"
            )
        return

    assets_root.mkdir(parents=True, exist_ok=True)
    results: dict[str, str] = {}
    failures: list[str] = []
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(download_one, spec, assets_root, timeout): spec.object_id
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
        raise RuntimeError(
            "Failed to download DTC assets (the URLs may have expired):\n- "
            + "\n- ".join(failures)
        )
    print(
        f"Downloaded {sum(result == 'downloaded' for result in results.values())} "
        f"objects to {assets_root / SOURCE_NAME} "
        f"({sum(result == 'cached' for result in results.values())} already verified)"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subset", type=Path, default=DEFAULT_SUBSET)
    parser.add_argument(
        "--download-urls-json",
        required=True,
        type=Path,
        help="Temporary URL manifest downloaded from the DTC website after licensing",
    )
    parser.add_argument(
        "--assets-root",
        type=Path,
        default=DEFAULT_ASSETS_ROOT,
        help="Canonical source-asset root; DTC files go to data/assets/dtc/<id>.glb",
    )
    parser.add_argument("--release", default=DEFAULT_RELEASE)
    parser.add_argument("--asset-key", default=DEFAULT_ASSET_KEY)
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
        download_urls_path=arguments.download_urls_json.expanduser().resolve(),
        assets_root=arguments.assets_root.expanduser().resolve(),
        release=arguments.release,
        asset_key=arguments.asset_key,
        workers=arguments.workers,
        timeout=arguments.timeout,
        dry_run=arguments.dry_run,
    )
