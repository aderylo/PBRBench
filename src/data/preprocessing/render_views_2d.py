"""Render registered 2D evaluation views from textured 3D assets."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import hydra
import rootutils
from omegaconf import DictConfig, OmegaConf

rootutils.setup_root(__file__, indicator=".project_root", pythonpath=True)

from src.data.preprocessing._common import (  # noqa: E402
    asset_path,
    resolve_lights,
    resolved,
    selected_objects,
    update_dataset_index,
    views,
    write_json,
)


def build_job(config: DictConfig, object_id: str, asset_path: Path) -> dict:
    output_dir = Path(config.data.output_root) / object_id / "2d"
    return {
        "schema_version": 2,
        "representation": "2d",
        "object_id": object_id,
        "asset_path": str(asset_path.resolve()),
        "output_dir": str(output_dir.resolve()),
        "views": views(config),
        "camera": resolved(config.camera),
        "lights": resolve_lights(config),
        "renderer": resolved(config.renderer),
        "overwrite": bool(config.overwrite),
    }


def rebuild_manifest(output_root: Path) -> None:
    rows = []
    for metadata_path in sorted(output_root.glob("*/2d/view_*/metadata.json")):
        metadata = json.loads(metadata_path.read_text())
        for light in metadata["lights"]:
            rows.append(
                {
                    "representation": "2d",
                    "object_id": metadata["object_id"],
                    "view_id": metadata["view_id"],
                    "light_id": light["id"],
                    "view_path": str(metadata_path.parent.relative_to(output_root)),
                    "rgb": str(
                        (
                            metadata_path.parent / "rgb" / f"{light['id']}.png"
                        ).relative_to(output_root)
                    ),
                }
            )
    output_root.mkdir(parents=True, exist_ok=True)
    manifest = output_root / "manifest_2d.jsonl"
    manifest.write_text("".join(json.dumps(row) + "\n" for row in rows))


@hydra.main(
    version_base=None, config_path="../../../configs", config_name="render_views_2d"
)
def main(config: DictConfig) -> None:
    object_ids = selected_objects(config)

    helper = Path(__file__).with_name("_render_views_2d.py")
    jobs_dir = Path(config.data.output_root) / ".jobs" / "2d"
    jobs_dir.mkdir(parents=True, exist_ok=True)

    for index, object_id in enumerate(object_ids, start=1):
        job = build_job(config, object_id, asset_path(config, object_id))
        job_path = jobs_dir / f"{object_id}.json"
        write_json(job_path, job)
        command = [
            str(config.renderer.executable),
            "--background",
            "--python",
            str(helper),
            "--",
            "--job",
            str(job_path.resolve()),
        ]
        print(f"[{index}/{len(object_ids)}] {object_id}")
        if config.dry_run:
            print(" ".join(command))
        else:
            subprocess.run(command, check=True)

    if not config.dry_run:
        output_root = Path(config.data.output_root)
        write_json(
            output_root / "dataset_2d.json",
            {
                "schema_version": 2,
                "dataset": str(config.data.name),
                "split": str(config.split),
                "representation": "2d",
                "channels": [
                    "rgb",
                    "albedo",
                    "roughness",
                    "metallic",
                    "normal",
                    "depth",
                    "mask",
                ],
                "normal_space": "camera",
                "resolved_config": OmegaConf.to_container(config, resolve=True),
            },
        )
        rebuild_manifest(output_root)
        update_dataset_index(output_root, str(config.data.name), str(config.split))


if __name__ == "__main__":
    main()
