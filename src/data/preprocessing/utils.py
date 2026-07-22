"""Shared preparation helpers and serializable Blender job types."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from omegaconf import DictConfig


def load_subset(path: Path) -> list[str]:
    """Load benchmark object IDs without requiring Blender packages."""
    import yaml

    payload = yaml.safe_load(path.read_text()) or {}
    objects = payload.get("objects", [])
    if not objects:
        raise ValueError(f"Object subset is empty: {path}")
    return [str(item["id"]) for item in objects]


def write_json(path: Path, payload: dict[str, Any]) -> None:
    """Write a readable JSON payload, creating its parent directory."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n")


def resolve_lights(config: "DictConfig") -> list[dict[str, Any]]:
    """Resolve configured HDR environment maps into Blender-ready records."""
    lights = []
    for envmap_id in config.lighting.envmaps:
        filename = (
            f"{envmap_id}_{config.lighting.resolution}."
            f"{config.lighting.format}"
        )
        path = Path(config.lighting.root) / filename
        if not path.is_file():
            raise FileNotFoundError(
                f"Missing HDRI {path}; run scripts/download_polyhaven_envmaps.py first"
            )
        lights.append(
            {
                "id": str(envmap_id),
                "path": str(path.resolve()),
                "rotation_deg": float(config.lighting.rotation_deg),
                "strength": float(config.lighting.strength),
            }
        )
    return lights


def selected_objects(config: "DictConfig") -> list[str]:
    """Select the requested object IDs in stable subset order."""
    object_ids = load_subset(Path(config.subset_file))
    if config.object_ids:
        requested = {str(item) for item in config.object_ids}
        object_ids = [object_id for object_id in object_ids if object_id in requested]
    if config.max_objects is not None:
        object_ids = object_ids[: int(config.max_objects)]
    return object_ids


def asset_path(config: "DictConfig", object_id: str) -> Path:
    """Resolve and validate one configured source asset path."""
    path = Path(config.source.root) / str(config.source.asset_template).format(
        object_id=object_id
    )
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def orbit_views(config: "DictConfig") -> list[dict[str, Any]]:
    """Expand the configured orbit into canonical view IDs."""
    return [
        {"id": f"view_{index:02d}", "yaw_deg": float(yaw)}
        for index, yaw in enumerate(config.views.yaw_degrees)
    ]


def resolved(value: "DictConfig") -> dict[str, Any]:
    """Resolve an OmegaConf node only when running in the project environment."""
    from omegaconf import OmegaConf

    return OmegaConf.to_container(value, resolve=True)


@dataclass(frozen=True)
class ViewSpec:
    """One canonical camera orbit view."""

    id: str
    yaw_deg: float

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ViewSpec":
        return cls(id=str(payload["id"]), yaw_deg=float(payload["yaw_deg"]))


@dataclass(frozen=True)
class CameraSpec:
    """Shared camera settings for all views of an asset."""

    elevation_deg: float
    distance: float
    focal_length_mm: float
    sensor_width_mm: float

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "CameraSpec":
        return cls(
            elevation_deg=float(payload["elevation_deg"]),
            distance=float(payload["distance"]),
            focal_length_mm=float(payload["focal_length_mm"]),
            sensor_width_mm=float(payload["sensor_width_mm"]),
        )


@dataclass(frozen=True)
class LightSpec:
    """One HDR environment used to render an RGB observation."""

    id: str
    path: str
    rotation_deg: float
    strength: float

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "LightSpec":
        return cls(
            id=str(payload["id"]),
            path=str(payload["path"]),
            rotation_deg=float(payload["rotation_deg"]),
            strength=float(payload["strength"]),
        )


@dataclass(frozen=True)
class RendererSpec:
    """The renderer options needed by Blender-side rendering helpers."""

    resolution: int
    samples_per_pixel: int
    denoise: bool
    device: str
    transparent_background: bool
    texture_resolution: int
    bake_margin: int

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "RendererSpec":
        return cls(
            resolution=int(payload["resolution"]),
            samples_per_pixel=int(payload["samples_per_pixel"]),
            denoise=bool(payload["denoise"]),
            device=str(payload["device"]),
            transparent_background=bool(payload.get("transparent_background", True)),
            texture_resolution=int(payload.get("texture_resolution", payload["resolution"])),
            bake_margin=int(payload.get("bake_margin", 16)),
        )


@dataclass(frozen=True)
class RenderJob:
    """All information Blender needs to render one object's 2D observations."""

    object_id: str
    asset_path: str
    output_dir: str
    views: tuple[ViewSpec, ...]
    camera: CameraSpec
    lights: tuple[LightSpec, ...]
    renderer: RendererSpec
    overwrite: bool
    min_foreground_pixels: int

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "RenderJob":
        return cls(
            object_id=str(payload["object_id"]),
            asset_path=str(payload["asset_path"]),
            output_dir=str(payload["output_dir"]),
            views=tuple(ViewSpec.from_dict(item) for item in payload["views"]),
            camera=CameraSpec.from_dict(payload["camera"]),
            lights=tuple(LightSpec.from_dict(item) for item in payload["lights"]),
            renderer=RendererSpec.from_dict(payload["renderer"]),
            overwrite=bool(payload["overwrite"]),
            min_foreground_pixels=int(payload["min_foreground_pixels"]),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class BakeJob:
    """All information Blender needs to bake one object's appearance textures."""

    object_id: str
    asset_path: str
    output_dir: str
    lights: tuple[LightSpec, ...]
    renderer: RendererSpec
    overwrite: bool

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "BakeJob":
        return cls(
            object_id=str(payload["object_id"]),
            asset_path=str(payload["asset_path"]),
            output_dir=str(payload["output_dir"]),
            lights=tuple(LightSpec.from_dict(item) for item in payload["lights"]),
            renderer=RendererSpec.from_dict(payload["renderer"]),
            overwrite=bool(payload["overwrite"]),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class CameraMetadata:
    """Camera calibration and pose needed to reproduce a prepared view."""

    resolution: tuple[int, int]
    intrinsics: list[list[float]]
    camera_to_world: list[list[float]]


@dataclass(frozen=True)
class ViewMetadata:
    """Non-derivable information required to use or relight a rendered view."""

    asset_path: str
    camera: CameraMetadata
    normalization_source_to_world: list[list[float]]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ObjectMetadata:
    """Non-derivable information required to use one prepared 3D object."""

    asset_path: str
    normalization_source_to_world: list[list[float]]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
