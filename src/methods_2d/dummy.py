"""Dummy screen-space material estimator.

Trivial baseline that emits zero roughness and zero metallic everywhere and
copies the input RGB observation as the albedo prediction. It needs no
weights, no GPU, and no upstream repository; it exists to smoke-test dataset
iteration and the inference pipeline.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm.auto import tqdm

from src.data.pbr_estimation_dataset_2d import PBREstimationSample2D
from src.methods_2d.base import BaseMaterialEstimator2D, Prediction2D


class Dummy2D(BaseMaterialEstimator2D):
    """Zero roughness/metallic, input RGB as albedo."""

    def setup(self) -> None:
        """Nothing to load for the dummy baseline."""

    def predict(
        self,
        samples: Sequence[PBREstimationSample2D],
        output_dir: Path,
    ) -> Mapping[str, Prediction2D]:
        if not samples:
            return {}

        outputs: dict[str, Prediction2D] = {}
        for sample in tqdm(
            samples,
            desc=f"Dummy 2D [{len(samples)} samples]",
            unit="sample",
        ):
            with Image.open(sample.rgb) as rgb_file:
                albedo = rgb_file.convert("RGB")
            zeros = Image.fromarray(
                np.zeros(albedo.size[::-1], dtype=np.uint8), mode="L"
            )

            outputs[sample.sample_id] = Prediction2D(
                albedo=albedo,
                roughness=zeros,
                metallic=zeros,
            ).save(save_dir=output_dir / sample.sample_id, mark_success=True)
        return outputs