"""Dummy 3D material estimator.

Trivial baseline that copies the baked UV texture atlas as the albedo
prediction and emits zero roughness and zero metallic everywhere. The
textures live directly on the sample mesh's own UV layout, so no remapping
is needed and the prediction is aligned to the canonical benchmark UVs by
copying the native GLB. It needs no weights, no GPU, and no upstream
repository; it exists to smoke-test dataset iteration and the inference
pipeline.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
from tqdm.auto import tqdm

from src.data.pbr_estimation_dataset_3d import PBREstimationSample3D
from src.methods_3d.base import BaseMaterialEstimator3D, Prediction3D
from src.utils.glb import create_textured_glb


class Dummy3D(BaseMaterialEstimator3D):
    """Zero roughness/metallic, baked texture as albedo."""

    uv_correspondence = "identity"
    texture_size = 1024

    def __init__(
        self,
        *,
        name: str = "dummy",
        project_root: str | Path = ".",
        texture_size: int = 1024,
        **kwargs: Any,
    ) -> None:
        super().__init__(name=name, project_root=project_root)
        self.texture_size = int(texture_size)

    def setup(self) -> None:
        """Nothing to load for the dummy baseline."""

    def teardown(self) -> None:
        """Nothing to release for the dummy baseline."""

    def predict_over_dataset(
        self,
        samples: Sequence[PBREstimationSample3D],
        output_dir: str | Path,
    ) -> Iterator[Prediction3D]:
        output_path = Path(output_dir)
        for sample in tqdm(
            samples,
            desc=f"Dummy 3D [{len(samples)} samples]",
            unit="sample",
        ):
            yield self._predict_sample(sample, output_path / sample.sample_id)

    def _predict_sample(
        self,
        sample: PBREstimationSample3D,
        sample_dir: Path,
    ) -> Prediction3D:
        sample_dir.mkdir(parents=True, exist_ok=True)

        with Image.open(sample.baked_texture) as baked_file:
            albedo = baked_file.convert("RGB")
        zeros = Image.fromarray(
            np.zeros(albedo.size[::-1], dtype=np.uint8), mode="L"
        )

        mesh_path = create_textured_glb(
            sample.mesh_path,
            {
                "albedo": albedo,
                "roughness": zeros,
                "metallic": zeros,
            },
            sample_dir / "mesh.glb",
        )

        return Prediction3D(
            sample_id=sample.sample_id,
            pbr_asset_glb=mesh_path,
        )