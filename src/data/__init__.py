"""Canonical benchmark datasets and schemas."""

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from src.data.envmaps import EnvMapDataset, EnvMapSpec
    from src.data.pbr_estimation_dataset_2d import (
        PBREstimationDataset2D,
        PBREstimationSample2D,
        ViewMetadata,
    )
    from src.data.pbr_estimation_dataset_3d import (
        MultiSourcePBREstimationDataset3D,
        PBREstimationDataset3D,
        PBREstimationSample3D,
    )

__all__ = [
    "EnvMapDataset",
    "EnvMapSpec",
    "PBREstimationDataset2D",
    "PBREstimationSample2D",
    "ViewMetadata",
    "PBREstimationDataset3D",
    "PBREstimationSample3D",
    "MultiSourcePBREstimationDataset3D",
]


def __getattr__(name: str) -> Any:
    if name in ("EnvMapDataset", "EnvMapSpec"):
        import src.data.envmaps as mod

        return getattr(mod, name)
    if name in ("PBREstimationDataset2D", "PBREstimationSample2D", "ViewMetadata"):
        import src.data.pbr_estimation_dataset_2d as mod

        return getattr(mod, name)
    if name in ("PBREstimationDataset3D", "PBREstimationSample3D", "MultiSourcePBREstimationDataset3D"):
        import src.data.pbr_estimation_dataset_3d as mod

        return getattr(mod, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
