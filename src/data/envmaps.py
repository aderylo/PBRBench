"""Dataset/collection for HDR environment maps used in relighting evaluation."""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _resolve_path(path: str | Path) -> Path:
    p = Path(path)
    return p.resolve() if p.is_absolute() else (PROJECT_ROOT / p).resolve()


@dataclass(frozen=True)
class EnvMapSpec:
    """Specification of a single HDR environment map."""

    id: str
    path: Path
    rotation_deg: float = 0.0
    strength: float = 1.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "path": str(self.path.resolve()),
            "rotation_deg": self.rotation_deg,
            "strength": self.strength,
        }


class EnvMapDataset(Sequence[EnvMapSpec]):
    """Discovers and provides HDR environment maps configured for evaluation."""

    def __init__(
        self,
        root: str | Path,
        envmaps: Sequence[str],
        resolution: str = "1k",
        format: str = "exr",
        rotation_deg: float = 0.0,
        strength: float = 1.0,
    ) -> None:
        self.root = _resolve_path(root)
        self.resolution = str(resolution)
        self.format = str(format)
        self.rotation_deg = float(rotation_deg)
        self.strength = float(strength)

        self._specs: list[EnvMapSpec] = []
        self._spec_map: dict[str, EnvMapSpec] = {}

        for envmap_id in envmaps:
            filename = f"{envmap_id}_{self.resolution}.{self.format}"
            path = self.root / filename
            if not path.is_file():
                raise FileNotFoundError(
                    f"Missing HDRI environment map {path}; please check {self.root} or run download script"
                )
            spec = EnvMapSpec(
                id=str(envmap_id),
                path=path,
                rotation_deg=self.rotation_deg,
                strength=self.strength,
            )
            self._specs.append(spec)
            self._spec_map[spec.id] = spec

    def get(self, envmap_id: str, default: EnvMapSpec | None = None) -> EnvMapSpec | None:
        """Get an environment map spec by its ID."""
        return self._spec_map.get(envmap_id, default)

    def filter(self, selected_ids: Sequence[str]) -> EnvMapDataset:
        """Return a subset of environment maps matching selected_ids."""
        selected_set = {str(item) for item in selected_ids}
        missing = selected_set - set(self._spec_map.keys())
        if missing:
            raise ValueError(f"Unknown target environment maps: {sorted(missing)}")

        filtered = EnvMapDataset.__new__(EnvMapDataset)
        filtered.root = self.root
        filtered.resolution = self.resolution
        filtered.format = self.format
        filtered.rotation_deg = self.rotation_deg
        filtered.strength = self.strength
        filtered._specs = [spec for spec in self._specs if spec.id in selected_set]
        filtered._spec_map = {spec.id: spec for spec in filtered._specs}
        return filtered

    def to_light_dicts(self) -> list[dict[str, Any]]:
        """Return list of light dictionaries formatted for renderer consumption."""
        return [spec.to_dict() for spec in self._specs]

    @property
    def ids(self) -> list[str]:
        """List of all environment map IDs."""
        return [spec.id for spec in self._specs]

    def __len__(self) -> int:
        return len(self._specs)

    def __getitem__(self, index: int) -> EnvMapSpec:
        return self._specs[index]

    def __iter__(self) -> Iterator[EnvMapSpec]:
        return iter(self._specs)

    def __contains__(self, envmap_id: object) -> bool:
        return envmap_id in self._spec_map
