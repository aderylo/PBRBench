"""Manifest-backed dataset for screen-space PBR estimation."""

from __future__ import annotations

import json
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class PBREstimationSample2D:
    """One benchmark sample containing input image and metadata."""

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
    """Load canonical 2D samples without eagerly decoding their images."""

    def __init__(
        self,
        root: str | Path,
        manifest: str | Path = "manifest_2d.jsonl",
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
        manifest_path = Path(manifest)
        self.manifest_path = (
            manifest_path if manifest_path.is_absolute() else self.root / manifest_path
        ).resolve()
        if not self.manifest_path.is_file():
            raise FileNotFoundError(f"2D manifest not found: {self.manifest_path}")

        if max_samples is not None and max_samples < 0:
            raise ValueError("max_samples must be non-negative or null")
        selected_objects = set(object_ids) if object_ids is not None else None
        selected_views = set(view_ids) if view_ids is not None else None
        selected_lights = set(light_ids) if light_ids is not None else None
        samples: list[PBREstimationSample2D] = []
        seen_ids: set[str] = set()

        if max_samples == 0:
            self._samples = ()
            return

        for line_number, line in enumerate(
            self.manifest_path.read_text().splitlines(), start=1
        ):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                object_id = str(row["object_id"])
                view_id = str(row["view_id"])
                light_id = str(row["light_id"])
            except (json.JSONDecodeError, KeyError) as error:
                raise ValueError(
                    f"Invalid row {line_number} in {self.manifest_path}: {error}"
                ) from error

            if row.get("representation") != "2d":
                raise ValueError(
                    f"Row {line_number} is not a 2D sample: "
                    f"{row.get('representation')!r}"
                )
            for label, value in (
                ("object_id", object_id),
                ("view_id", view_id),
                ("light_id", light_id),
            ):
                self._validate_identifier(value, label, line_number)

            if selected_objects is not None and object_id not in selected_objects:
                continue
            if selected_views is not None and view_id not in selected_views:
                continue
            if selected_lights is not None and light_id not in selected_lights:
                continue

            sample_id = f"{object_id}__{view_id}__{light_id}"
            if sample_id in seen_ids:
                raise ValueError(f"Duplicate sample_id in manifest: {sample_id}")
            seen_ids.add(sample_id)

            view_dir = self._resolve(row["view_path"])
            metadata_path = view_dir / "metadata.json"
            try:
                metadata = json.loads(metadata_path.read_text())
            except (FileNotFoundError, json.JSONDecodeError):
                metadata = {}

            rgb_path = self._resolve(row["rgb"])

            mask_path = view_dir / "mask.png"
            normal_path = view_dir / "normal.png"
            albedo_path = view_dir / "albedo.png"
            roughness_path = view_dir / "roughness.png"
            metallic_path = view_dir / "metallic.png"
            depth_path = view_dir / "depth.png"

            sample = PBREstimationSample2D(
                sample_id=sample_id,
                object_id=object_id,
                view_id=view_id,
                light_id=light_id,
                rgb=rgb_path,
                mask=mask_path if mask_path.is_file() else None,
                normal=normal_path if normal_path.is_file() else None,
                albedo=albedo_path if albedo_path.is_file() else None,
                roughness=roughness_path if roughness_path.is_file() else None,
                metallic=metallic_path if metallic_path.is_file() else None,
                depth=depth_path if depth_path.is_file() else None,
                metadata=metadata,
            )

            if validate_files and not rgb_path.is_file():
                raise FileNotFoundError(f"Missing input image for {sample_id}: {rgb_path}")

            samples.append(sample)
            if max_samples is not None and len(samples) >= max_samples:
                break

        self._samples = tuple(samples)

    @staticmethod
    def _validate_identifier(value: str, label: str, line_number: int) -> None:
        if not value or value in {".", ".."} or "/" in value or "\\" in value:
            raise ValueError(f"Unsafe {label} at manifest row {line_number}: {value!r}")

    def _resolve(self, value: str | Path) -> Path:
        path = (self.root / Path(value)).resolve()
        if not path.is_relative_to(self.root):
            raise ValueError(f"Manifest path escapes dataset root: {value}")
        return path

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, index: int) -> PBREstimationSample2D:
        return self._samples[index]

    def __iter__(self) -> Iterator[PBREstimationSample2D]:
        return iter(self._samples)
