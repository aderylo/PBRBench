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
