"""Directory-backed dataset for prepared 3D PBR observations."""

from __future__ import annotations

import json
from bisect import bisect_right
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from itertools import accumulate
from pathlib import Path
from typing import Any

import yaml
from hydra.utils import instantiate

PROJECT_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class PBREstimationSample3D:
    """One registered 3D observation and its canonical PBR references."""

    sample_id: str
    object_id: str
    baked_texture_id: str
    mesh_path: Path
    baked_texture: Path
    albedo: Path | None = None
    roughness: Path | None = None
    metallic: Path | None = None
    normal: Path | None = None
    uv_mask: Path | None = None
    reference_view: Path | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict, repr=False)
    source: str = ""

    @property
    def texture_id(self) -> str:
        return self.baked_texture_id

    def load_trimesh(self, process: bool = False) -> Any:
        import trimesh

        mesh = trimesh.load(str(self.mesh_path), process=process)
        if isinstance(mesh, trimesh.Scene):
            mesh = mesh.to_mesh()
        return mesh


class PBREstimationDataset3D(Sequence[PBREstimationSample3D]):
    """Discover completed 3D object directories deterministically."""

    def __init__(
        self,
        data_dir: str | Path,
        split_file: str | Path | None = None,
        limit: int | None = None,
        source: str = "",
    ) -> None:
        self.data_dir = self._resolve_path(data_dir)
        self.split_file = self._resolve_path(split_file) if split_file else None
        self.source = source or self.data_dir.name
        self.limit = limit

        if not self.data_dir.is_dir():
            raise FileNotFoundError(f"3D dataset directory not found: {self.data_dir}")
        if self.limit is not None and self.limit < 0:
            raise ValueError("limit must be non-negative or null")

        self._sample_map: dict[str, PBREstimationSample3D] = {}
        if self.limit == 0:
            self._samples: tuple[PBREstimationSample3D, ...] = ()
            return

        allowed = self._parse_split(self.split_file) if self.split_file else None
        samples: list[PBREstimationSample3D] = []

        for obj_dir in sorted(self.data_dir.iterdir()):
            if not obj_dir.is_dir() or obj_dir.name.startswith("."):
                continue

            object_id = obj_dir.name
            if allowed is not None and object_id not in allowed:
                continue

            dir_3d = obj_dir / "3d" if (obj_dir / "3d").is_dir() else obj_dir
            dir_2d = obj_dir / "2d" if (obj_dir / "2d").is_dir() else None

            mesh_path = dir_3d / "mesh.obj"
            if not mesh_path.is_file():
                continue

            ref_view_dir: Path | None = None
            light_set: set[str] | None = None

            if allowed is not None:
                view_spec = allowed[object_id]
                explicit_views = [v for v in view_spec if v != "*"]
                if explicit_views:
                    ref_view_id = explicit_views[0]
                    light_set = view_spec[ref_view_id]
                    if dir_2d is not None and (dir_2d / ref_view_id).is_dir():
                        ref_view_dir = dir_2d / ref_view_id
                else:
                    light_set = view_spec.get("*")

            metadata_path = dir_3d / "metadata.json"
            metadata = (
                json.loads(metadata_path.read_text())
                if metadata_path.is_file()
                else {}
            )

            uv_mask_path = self._optional_file(dir_3d / "uv_mask.png") or self._optional_file(
                dir_3d / "mask.png"
            )

            pbr_dir = dir_3d / "pbr" if (dir_3d / "pbr").is_dir() else dir_3d
            pbr_paths: dict[str, Path] = {}
            for channel in ("albedo", "roughness", "metallic", "normal"):
                for ext in (".png", ".jpg"):
                    cand = pbr_dir / f"{channel}{ext}"
                    if cand.is_file():
                        pbr_paths[channel] = cand
                        break

            tex_dir = dir_3d / "textures"
            tex_paths = (
                sorted(tex_dir.glob("*.png")) + sorted(tex_dir.glob("*.jpg"))
                if tex_dir.is_dir()
                else []
            )

            for tex_path in tex_paths:
                tex_id = tex_path.stem
                if light_set is not None and tex_id not in light_set:
                    continue

                sample_id = f"{self.source}__{object_id}__{tex_id}"
                if sample_id in self._sample_map:
                    raise ValueError(f"Duplicate sample_id: {sample_id}")

                ref_view_path: Path | None = None
                if ref_view_dir is not None:
                    for ext in (".png", ".jpg"):
                        cand = ref_view_dir / "rgb" / f"{tex_id}{ext}"
                        if cand.is_file():
                            ref_view_path = cand
                            break

                sample = PBREstimationSample3D(
                    sample_id=sample_id,
                    object_id=object_id,
                    baked_texture_id=tex_id,
                    mesh_path=mesh_path,
                    baked_texture=tex_path,
                    albedo=pbr_paths.get("albedo"),
                    roughness=pbr_paths.get("roughness"),
                    metallic=pbr_paths.get("metallic"),
                    normal=pbr_paths.get("normal"),
                    uv_mask=uv_mask_path,
                    reference_view=ref_view_path,
                    metadata=metadata,
                    source=self.source,
                )
                samples.append(sample)
                self._sample_map[sample_id] = sample
                if self.limit is not None and len(samples) >= self.limit:
                    self._samples = tuple(samples)
                    return

        self._samples = tuple(samples)

    @staticmethod
    def _resolve_path(path: str | Path) -> Path:
        p = Path(path)
        return p.resolve() if p.is_absolute() else (PROJECT_ROOT / p).resolve()

    @staticmethod
    def _parse_split(path: Path) -> dict[str, dict[str, set[str] | None]]:
        payload = yaml.safe_load(path.read_text()) or {}
        items = (
            payload
            if isinstance(payload, list)
            else (payload.get("samples") or payload.get("objects") or payload.get("ids") or [])
        )
        specs: dict[str, dict[str, set[str] | None]] = {}
        for item in items:
            if isinstance(item, dict):
                obj = item.get("object") or item.get("id")
                if not obj:
                    continue
                views = item.get("views")
                lights = item.get("lights")
                light_set = {str(v) for v in lights} if lights else None
                view_spec = specs.setdefault(str(obj), {})
                if views:
                    for view in views:
                        view_spec[str(view)] = light_set
                else:
                    view_spec["*"] = light_set
            else:
                specs[str(item)] = {"*": None}
        return specs

    @staticmethod
    def _optional_file(path: Path) -> Path | None:
        return path if path.is_file() else None

    def get_sample(self, sample_id: str) -> PBREstimationSample3D | None:
        """Get one sample by its canonical ID."""
        return self._sample_map.get(sample_id)

    def get(
        self, sample_id: str, default: PBREstimationSample3D | None = None
    ) -> PBREstimationSample3D | None:
        """Get one sample by its canonical ID with optional default."""
        return self._sample_map.get(sample_id, default)

    def __contains__(self, sample_id: object) -> bool:
        return sample_id in self._sample_map

    def keys(self) -> tuple[str, ...]:
        return tuple(self._sample_map.keys())

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, index: int) -> PBREstimationSample3D:
        return self._samples[index]

    def __iter__(self) -> Iterator[PBREstimationSample3D]:
        return iter(self._samples)


class MultiSourcePBREstimationDataset3D(Sequence[PBREstimationSample3D]):
    """Thin wrapper over independently configured sub-datasets."""

    def __init__(
        self,
        datasets: Sequence[Mapping[str, Any] | PBREstimationDataset3D],
        limit: int | None = None,
    ) -> None:
        if limit is not None and limit < 0:
            raise ValueError("limit must be non-negative or null")

        self._datasets: list[PBREstimationDataset3D] = []
        remaining = limit
        for cfg in datasets:
            if remaining is not None and remaining <= 0:
                break
            sub_dataset = (
                cfg if isinstance(cfg, PBREstimationDataset3D) else instantiate(cfg)
            )
            self._datasets.append(sub_dataset)
            if remaining is not None:
                remaining -= len(sub_dataset)

        self._samples = tuple(
            sample for dataset in self._datasets for sample in dataset
        )
        if limit is not None:
            self._samples = self._samples[:limit]
        self._offsets = list(accumulate(len(d) for d in self._datasets))
        self._sample_map = {sample.sample_id: sample for sample in self._samples}

    def get_sample(self, sample_id: str) -> PBREstimationSample3D | None:
        return self._sample_map.get(sample_id)

    def get(
        self, sample_id: str, default: PBREstimationSample3D | None = None
    ) -> PBREstimationSample3D | None:
        return self._sample_map.get(sample_id, default)

    def __contains__(self, sample_id: object) -> bool:
        return sample_id in self._sample_map

    def keys(self) -> tuple[str, ...]:
        return tuple(self._sample_map.keys())

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, index: int) -> PBREstimationSample3D:
        size = len(self)
        if index < 0:
            index += size
        if index < 0 or index >= size:
            raise IndexError(index)
        sub_index = bisect_right(self._offsets, index)
        if sub_index == len(self._datasets):
            sub_index -= 1
        prev = self._offsets[sub_index - 1] if sub_index > 0 else 0
        return self._datasets[sub_index][index - prev]

    def __iter__(self) -> Iterator[PBREstimationSample3D]:
        return iter(self._samples)
