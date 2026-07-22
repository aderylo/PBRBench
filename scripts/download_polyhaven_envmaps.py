"""Download the Poly Haven HDRIs named in a light config."""

import argparse
import os
from concurrent.futures import ThreadPoolExecutor

import requests
import yaml
from tqdm import tqdm


def get_file_url(asset_id, resolution="1k", format_pref="exr"):
    response = requests.get(f"https://api.polyhaven.com/files/{asset_id}", timeout=30)
    response.raise_for_status()
    resolutions = response.json().get("hdri", {})
    formats = resolutions.get(resolution, {})
    if format_pref in formats:
        return formats[format_pref]["url"]
    if "hdr" in formats:
        return formats["hdr"]["url"]
    raise ValueError(f"No {resolution} EXR/HDR file found for {asset_id}")


def download_file(url, save_path):
    if os.path.exists(save_path):
        return
    try:
        response = requests.get(url, stream=True, timeout=120)
        response.raise_for_status()
        with open(save_path, "wb") as output:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                output.write(chunk)
    except Exception:
        if os.path.exists(save_path):
            os.remove(save_path)
        raise


def process_asset(asset_id, output_dir, resolution, format_pref):
    url = get_file_url(asset_id, resolution, format_pref)
    download_file(url, os.path.join(output_dir, url.split("/")[-1]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="configs/data/preprocessing/lighting/polyhaven_source_v0.yaml",
    )
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    with open(args.config) as stream:
        config = yaml.safe_load(stream)

    output_dir = args.output_dir or config["root"]
    os.makedirs(output_dir, exist_ok=True)
    envmaps = config["envmaps"]
    resolution = config.get("resolution", "1k")
    format_pref = config.get("format", "exr")

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = [
            executor.submit(process_asset, asset_id, output_dir, resolution, format_pref)
            for asset_id in envmaps
        ]
        for future in tqdm(futures, desc=f"Downloading {resolution} HDRIs"):
            future.result()


if __name__ == "__main__":
    main()
