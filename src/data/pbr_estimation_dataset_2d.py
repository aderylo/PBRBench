"""Directory-backed dataset for prepared screen-space PBR observations."""

from __future__ import annotations

import json
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


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


class PBREstimationDataset2D(Sequence[PBREstimationSample2D]):
    """Discover completed ``<object>/<view>`` directories deterministically."""

    def __init__(
        self,
        root: str | Path,
        *,
        name: str = "pbr_estimation_2d",
        object_ids: Sequence[str] | None = None,
        view_ids: Sequence[str] | None = None,
        light_ids: Sequence[str] | None = None,
        max_samples: int | None = None,
        validate_files: bool = True,
    ) -> None:
        self.name = name
        self.root = Path(root).resolve()
        if not self.root.is_dir():
            raise FileNotFoundError(f"2D dataset root not found: {self.root}")
        if max_samples is not None and max_samples < 0:
            raise ValueError("max_samples must be non-negative or null")

        self._sample_map: dict[str, PBREstimationSample2D] = {}
        if max_samples == 0:
            self._samples = ()
            return

        selected_objects = set(object_ids) if object_ids is not None else None
        selected_views = set(view_ids) if view_ids is not None else None
        selected_lights = set(light_ids) if light_ids is not None else None
        samples: list[PBREstimationSample2D] = []

        for metadata_path in sorted(self.root.glob("*/view_*/metadata.json")):
            view_dir = metadata_path.parent
            object_id = view_dir.parent.name
            view_id = view_dir.name
            self._validate_identifier(object_id, "object_id", metadata_path)
            self._validate_identifier(view_id, "view_id", metadata_path)
            if selected_objects is not None and object_id not in selected_objects:
                continue
            if selected_views is not None and view_id not in selected_views:
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
                if selected_lights is not None and light_id not in selected_lights:
                    continue
                sample_id = f"{object_id}__{view_id}__{light_id}"
                if sample_id in self._sample_map:
                    raise ValueError(f"Duplicate sample_id: {sample_id}")
                if validate_files and not rgb_path.is_file():
                    raise FileNotFoundError(
                        f"Missing input image for {sample_id}: {rgb_path}"
                    )

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
                )
                samples.append(sample)
                self._sample_map[sample_id] = sample
                if max_samples is not None and len(samples) >= max_samples:
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
