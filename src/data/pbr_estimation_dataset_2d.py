"""Directory-backed dataset for prepared screen-space PBR observations."""

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


def _resolve_path(path: str | Path) -> Path:
    p = Path(path)
    return p.resolve() if p.is_absolute() else (PROJECT_ROOT / p).resolve()


def _parse_split(path: Path) -> dict[str, dict[str, set[str] | None]]:
    """Parse a split file into ``object -> {view | "*": lights}`` specs.

    The inner mapping keys are view ids; a ``"*"`` key means every view of the
    object is allowed.  The value is the set of allowed light ids, or ``None``
    when every light is allowed.
    """
    payload = yaml.safe_load(path.read_text()) or {}
    items = (
        payload
        if isinstance(payload, list)
        else (payload.get("samples") or payload.get("objects") or [])
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


@dataclass(frozen=True)
class ViewMetadata:
    """Camera viewpoint and object normalization parameters for rendering and relighting.

    Attributes:
        camera: Camera extrinsic pose (``camera_to_world``), intrinsic matrix
            (``intrinsics``), and render resolution (``resolution``).
        normalization_source_to_world: 4x4 matrix scaling and centering the raw
            mesh into the standardized unit bounding box.
        asset_path: Path to the raw source 3D asset file before benchmark normalization.
        recipe_hash: Optional hash of the render recipe used to generate this view.
    """

    camera: Mapping[str, Any]
    normalization_source_to_world: list[list[float]]
    asset_path: Path | None = None
    recipe_hash: str | None = None

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> ViewMetadata:
        asset = Path(data["asset_path"]) if data.get("asset_path") else None
        return cls(
            camera=data.get("camera", {}),
            normalization_source_to_world=data.get("normalization_source_to_world", []),
            asset_path=asset,
            recipe_hash=data.get("recipe_hash"),
        )


@dataclass(frozen=True)
class PBREstimationSample2D:
    """One registered RGB observation and its view-level PBR references."""

    sample_id: str
    object_id: str
    view_id: str
    light_id: str
    rgb: Path
    mask: Path | None = None
    normal: Path | None = None
    albedo: Path | None = None
    roughness: Path | None = None
    metallic: Path | None = None
    depth: Path | None = None
    view_metadata: ViewMetadata | None = None
    extra_metadata: Mapping[str, Any] = field(default_factory=dict, repr=False)
    source: str = ""


class PBREstimationDataset2D(Sequence[PBREstimationSample2D]):
    """Discover completed ``<object>/<view>`` directories deterministically."""

    def __init__(
        self,
        data_dir: str | Path,
        split_file: str | Path | None = None,
        limit: int | None = None,
        source: str = "",
    ) -> None:
        self.data_dir = _resolve_path(data_dir)
        self.split_file = _resolve_path(split_file) if split_file else None
        self.source = source or self.data_dir.name
        self.limit = limit

        if not self.data_dir.is_dir():
            raise FileNotFoundError(f"2D dataset directory not found: {self.data_dir}")
        if self.limit is not None and self.limit < 0:
            raise ValueError("limit must be non-negative or null")

        self._sample_map: dict[str, PBREstimationSample2D] = {}
        if self.limit == 0:
            self._samples = ()
            return

        allowed = _parse_split(self.split_file) if self.split_file else None
        samples: list[PBREstimationSample2D] = []

        for metadata_path in sorted(self.data_dir.glob("*/view_*/metadata.json")):
            view_dir = metadata_path.parent
            object_id = view_dir.parent.name
            view_id = view_dir.name
            self._validate_identifier(object_id, "object_id", metadata_path)
            self._validate_identifier(view_id, "view_id", metadata_path)

            light_set: set[str] | None = None
            if allowed is not None:
                view_spec = allowed.get(object_id)
                if view_spec is None:
                    continue
                if "*" in view_spec:
                    light_set = view_spec["*"]
                elif view_id in view_spec:
                    light_set = view_spec[view_id]
                else:
                    continue

            try:
                metadata = json.loads(metadata_path.read_text())
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid view metadata: {metadata_path}") from error

            rgb_dir = view_dir / "rgb"
            rgb_paths = sorted(rgb_dir.glob("*.png")) if rgb_dir.is_dir() else []
            if not rgb_paths:
                raise FileNotFoundError(
                    f"Completed view has no RGB observations: {view_dir}"
                )

            for rgb_path in rgb_paths:
                light_id = rgb_path.stem
                self._validate_identifier(light_id, "light_id", rgb_path)
                if light_set is not None and light_id not in light_set:
                    continue
                sample_id = f"{self.source}__{object_id}__{view_id}__{light_id}"
                if sample_id in self._sample_map:
                    raise ValueError(f"Duplicate sample_id: {sample_id}")

                sample = PBREstimationSample2D(
                    sample_id=sample_id,
                    object_id=object_id,
                    view_id=view_id,
                    light_id=light_id,
                    rgb=rgb_path,
                    mask=self._optional_file(view_dir / "mask.png"),
                    normal=self._optional_file(view_dir / "normal.png"),
                    albedo=self._optional_file(view_dir / "albedo.png"),
                    roughness=self._optional_file(view_dir / "roughness.png"),
                    metallic=self._optional_file(view_dir / "metallic.png"),
                    depth=self._optional_file(view_dir / "depth.png"),
                    view_metadata=ViewMetadata.from_dict(metadata),
                    extra_metadata=metadata,
                    source=self.source,
                )
                samples.append(sample)
                self._sample_map[sample_id] = sample
                if self.limit is not None and len(samples) >= self.limit:
                    self._samples = tuple(samples)
                    return

        self._samples = tuple(samples)

    @staticmethod
    def _optional_file(path: Path) -> Path | None:
        return path if path.is_file() else None

    @staticmethod
    def _validate_identifier(value: str, label: str, source: Path) -> None:
        if not value or value in {".", ".."} or "/" in value or "\\" in value:
            raise ValueError(f"Unsafe {label} in {source}: {value!r}")

    def get_sample(self, sample_id: str) -> PBREstimationSample2D | None:
        """Get one sample by its canonical ID."""
        return self._sample_map.get(sample_id)

    def get(
        self, sample_id: str, default: PBREstimationSample2D | None = None
    ) -> PBREstimationSample2D | None:
        """Get one sample by its canonical ID with optional default."""
        return self._sample_map.get(sample_id, default)

    def __contains__(self, sample_id: object) -> bool:
        return sample_id in self._sample_map

    def keys(self) -> tuple[str, ...]:
        return tuple(self._sample_map.keys())

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, index: int) -> PBREstimationSample2D:
        return self._samples[index]

    def __iter__(self) -> Iterator[PBREstimationSample2D]:
        return iter(self._samples)


class MultiSourcePBREstimationDataset2D(Sequence[PBREstimationSample2D]):
    """Thin wrapper over independently configured sub-datasets.

    Each entry in ``datasets`` is a per-source dataset config (carrying its own
    ``_target_``); they are instantiated one by one in order, and index lookups
    are routed to the owning sub-dataset via cumulative offsets.
    """

    def __init__(
        self,
        datasets: Sequence[Mapping[str, Any] | PBREstimationDataset2D],
        limit: int | None = None,
    ) -> None:
        if limit is not None and limit < 0:
            raise ValueError("limit must be non-negative or null")

        self._datasets: list[PBREstimationDataset2D] = []
        remaining = limit
        for cfg in datasets:
            if remaining is not None and remaining <= 0:
                break
            sub_dataset = (
                cfg if isinstance(cfg, PBREstimationDataset2D) else instantiate(cfg)
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

    def get_sample(self, sample_id: str) -> PBREstimationSample2D | None:
        return self._sample_map.get(sample_id)

    def get(
        self, sample_id: str, default: PBREstimationSample2D | None = None
    ) -> PBREstimationSample2D | None:
        return self._sample_map.get(sample_id, default)

    def __contains__(self, sample_id: object) -> bool:
        return sample_id in self._sample_map

    def keys(self) -> tuple[str, ...]:
        return tuple(self._sample_map.keys())

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, index: int) -> PBREstimationSample2D:
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

    def __iter__(self) -> Iterator[PBREstimationSample2D]:
        return iter(self._samples)

