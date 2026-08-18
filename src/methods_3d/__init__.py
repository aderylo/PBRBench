from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from src.methods_3d.base import BaseMaterialEstimator3D, Prediction3D
    from src.methods_3d.dummy import Dummy3D
    from src.methods_3d.hunyuan3d import Hunyuan3DEstimator3D
    from src.methods_3d.material_anything import MaterialAnythingEstimator3D
    from src.methods_3d.trellis2 import Trellis2Estimator3D

__all__ = [
    "BaseMaterialEstimator3D",
    "Prediction3D",
    "Dummy3D",
    "Hunyuan3DEstimator3D",
    "MaterialAnythingEstimator3D",
    "Trellis2Estimator3D",
]


def __getattr__(name: str) -> Any:
    if name in ("BaseMaterialEstimator3D", "Prediction3D"):
        import src.methods_3d.base as mod

        return getattr(mod, name)
    if name == "Dummy3D":
        import src.methods_3d.dummy as mod

        return getattr(mod, name)
    if name == "Hunyuan3DEstimator3D":
        import src.methods_3d.hunyuan3d as mod

        return getattr(mod, name)
    if name == "MaterialAnythingEstimator3D":
        import src.methods_3d.material_anything as mod

        return getattr(mod, name)
    if name == "Trellis2Estimator3D":
        import src.methods_3d.trellis2 as mod

        return getattr(mod, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
