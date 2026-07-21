"""Small orchestration helpers shared by the 2D and 3D render launchers."""

from __future__ import annotations

import json
from pathlib import Path

import yaml
from omegaconf import DictConfig, OmegaConf


def load_split(path: Path, split: str) -> list[str]:
    payload = yaml.safe_load(path.read_text()) or {}
    if split not in payload:
        raise KeyError(f"Unknown split '{split}' in {path}")
    return [str(item["id"]) for item in payload[split]]


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n")


def update_dataset_index(output_root: Path, dataset: str, split: str) -> None:
    """Describe the representations currently present below one dataset root."""
    representations = [
        name
        for name in ("2d", "3d")
        if any(output_root.glob(f"*/{name}/view_*/metadata.json"))
    ]
    write_json(
        output_root / "dataset.json",
        {
            "schema_version": 2,
            "dataset": dataset,
            "split": split,
            "layout": "<object_id>/<representation>/<view_id>",
            "representations": representations,
        },
    )


def resolve_lights(config: DictConfig) -> list[dict]:
    lights = []
    for envmap_id in config.lights.envmaps:
        filename = f"{envmap_id}_{config.lights.resolution}.{config.lights.format}"
        path = Path(config.lights.root) / filename
        if not path.is_file():
            raise FileNotFoundError(
                f"Missing HDRI {path}; run scripts/download_polyhaven_envmaps.py first"
            )
        lights.append(
            {
                "id": str(envmap_id),
                "path": str(path.resolve()),
                "rotation_deg": float(config.lights.rotation_deg),
                "strength": float(config.lights.strength),
            }
        )
    return lights


def selected_objects(config: DictConfig) -> list[str]:
    object_ids = load_split(Path(config.data.split_file), str(config.split))
    if config.object_ids:
        requested = {str(item) for item in config.object_ids}
        object_ids = [object_id for object_id in object_ids if object_id in requested]
    if config.max_objects is not None:
        object_ids = object_ids[: int(config.max_objects)]
    return object_ids


def asset_path(config: DictConfig, object_id: str) -> Path:
    path = Path(config.data.root) / str(config.data.path_template).format(id=object_id)
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def views(config: DictConfig) -> list[dict]:
    return [
        {"id": f"view_{index:02d}", "yaw_deg": float(yaw)}
        for index, yaw in enumerate(config.camera.yaw_degrees)
    ]


def resolved(value: DictConfig) -> dict:
    return OmegaConf.to_container(value, resolve=True)
