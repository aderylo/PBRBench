"""Directory-backed dataset for prepared screen-space PBR observations."""

from __future__ import annotations

import json
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _resolve_path(path: str | Path) -> Path:
    p = Path(path)
    return p.resolve() if p.is_absolute() else (PROJECT_ROOT / p).resolve()


def _parse_split(path: Path) -> dict[str, set[str] | None]:
    payload = yaml.safe_load(path.read_text()) or {}
    items = (
        payload
        if isinstance(payload, list)
        else (payload.get("samples") or payload.get("objects") or [])
    )
    specs: dict[str, set[str] | None] = {}
    for item in items:
        if isinstance(item, dict):
            obj = item.get("object") or item.get("id")
            if not obj:
                continue
            views = item.get("views")
            specs[str(obj)] = {str(v) for v in views} if views else None
        else:
            specs[str(item)] = None
    return specs


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
    metadata: Mapping[str, Any] = field(default_factory=dict, repr=False)
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
            if allowed is not None:
                if object_id not in allowed:
                    continue
                allowed_views = allowed[object_id]
                if allowed_views is not None and view_id not in allowed_views:
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
    def _optional_file(path: Path) -> Path | None:
        return path if path.is_file() else None

    @staticmethod
    def _validate_identifier(value: str, label: str, source: Path) -> None:
        if not value or value in {".", ".."} or "/" in value or "\\" in value:
            raise ValueError(f"Unsafe {label} in {source}: {value!r}")

    def get_sample(self, sample_id: str) -> PBREstimationSample2D | None:
        """Get one sample by its canonical ID."""
        return self._sample_map.get(sample_id)

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, index: int) -> PBREstimationSample2D:
        return self._samples[index]

    def __iter__(self) -> Iterator[PBREstimationSample2D]:
        return iter(self._samples)


class MultiSourcePBREstimationDataset2D(Sequence[PBREstimationSample2D]):
    """Discover completed observation directories across multiple data sources."""

    def __init__(
        self,
        sources: Mapping[str, str | Path | Mapping[str, Any]],
        split_file: str | Path | None = None,
        limit: int | None = None,
    ) -> None:
        self.limit = limit
        self._datasets: list[PBREstimationDataset2D] = []
        remaining = limit

        for name, cfg in sources.items():
            if remaining is not None and remaining <= 0:
                break
            if isinstance(cfg, (str, Path)):
                dir_path = Path(cfg)
                kwargs: dict[str, Any] = {}
            elif isinstance(cfg, Mapping):
                dir_path = Path(cfg["data_dir"])
                kwargs = {k: v for k, v in cfg.items() if k != "data_dir"}
            else:
                raise TypeError(f"Invalid source config for {name}: {cfg}")

            sub_ds = PBREstimationDataset2D(
                data_dir=dir_path,
                source=name,
                split_file=kwargs.get("split_file", split_file),
                limit=kwargs.get("limit", remaining),
            )
            self._datasets.append(sub_ds)
            if remaining is not None:
                remaining -= len(sub_ds)

        self._samples = tuple(
            sample for dataset in self._datasets for sample in dataset
        )
        if limit is not None:
            self._samples = self._samples[:limit]
        self._sample_map = {sample.sample_id: sample for sample in self._samples}

    def get_sample(self, sample_id: str) -> PBREstimationSample2D | None:
        return self._sample_map.get(sample_id)

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, index: int) -> PBREstimationSample2D:
        return self._samples[index]

    def __iter__(self) -> Iterator[PBREstimationSample2D]:
        return iter(self._samples)
