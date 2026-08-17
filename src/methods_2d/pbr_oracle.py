"""PBROracle diffuse-albedo estimation from matched relightings.

This first benchmark version is deliberately physics-only. It renders
controlled full-PBR relightings and matched neutral-Lambertian shading proxies,
divides transport out in linear RGB, and aggregates the normalized observations
with a per-channel soft-weighted median in log space.
"""

from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
from tqdm.auto import tqdm

from src.data.pbr_estimation_dataset_2d import PBREstimationSample2D
from src.methods_2d.base import BaseMaterialEstimator2D, Prediction2D
from third_party.pbr_oracle import (
    BlenderRelightingOracle,
    BlenderShadingOracle,
    LightSpec,
    PreparedRelightingOracle,
    estimate_diffuse_albedo,
)


class PBROracle2D(BaseMaterialEstimator2D):
    """Estimate screen-space diffuse albedo using a virtual light stage."""

    def __init__(
        self,
        *,
        lights: Sequence[Mapping[str, Any]],
        relighting_oracle: Mapping[str, Any],
        shading_oracle: Mapping[str, Any],
        proxy_reflectance: float = 0.18,
        energy_floor: float = 0.003,
        clipping_start: float = 0.90,
        minimum_clipping_weight: float = 0.02,
        max_ratio: float = 8.0,
        log_epsilon: float = 1.0e-4,
        dummy_roughness: float = 0.5,
        dummy_metallic: float = 0.0,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.light_payloads = [dict(item) for item in lights]
        self.relighting_config = dict(relighting_oracle)
        self.shading_config = dict(shading_oracle)
        self.proxy_reflectance = float(proxy_reflectance)
        self.energy_floor = float(energy_floor)
        self.clipping_start = float(clipping_start)
        self.minimum_clipping_weight = float(minimum_clipping_weight)
        self.max_ratio = float(max_ratio)
        self.log_epsilon = float(log_epsilon)
        self.dummy_roughness = float(dummy_roughness)
        self.dummy_metallic = float(dummy_metallic)
        self.lights: list[LightSpec] = []
        self.relighting_backend: (
            PreparedRelightingOracle | BlenderRelightingOracle | None
        ) = None
        self.shading_backend: BlenderShadingOracle | None = None

    def setup(self) -> None:
        super().setup()
        if not 0.0 <= self.dummy_roughness <= 1.0:
            raise ValueError("dummy_roughness must be in [0, 1]")
        if not 0.0 <= self.dummy_metallic <= 1.0:
            raise ValueError("dummy_metallic must be in [0, 1]")
        backend = str(self.relighting_config.get("backend", "prepared"))
        if backend not in {"prepared", "blender"}:
            raise ValueError(f"Unsupported relighting oracle backend: {backend!r}")
        self.lights = [
            LightSpec.from_mapping(payload, project_root=self.project_root)
            for payload in self.light_payloads
        ]
        if len({light.id for light in self.lights}) != len(self.lights):
            raise ValueError("PBROracle light ids must be unique")
        if backend == "prepared":
            self.relighting_backend = PreparedRelightingOracle()
        else:
            self.relighting_backend = BlenderRelightingOracle(
                blender=str(self.relighting_config.get("blender", "blender")),
                samples_per_pixel=int(
                    self.relighting_config.get("samples_per_pixel", 32)
                ),
                denoise=bool(self.relighting_config.get("denoise", True)),
                device=str(self.relighting_config.get("device", "cuda")),
                overwrite=bool(self.relighting_config.get("overwrite", False)),
            )
        self.shading_backend = BlenderShadingOracle(
            blender=str(self.shading_config.get("blender", "blender")),
            samples_per_pixel=int(
                self.shading_config.get("samples_per_pixel", 32)
            ),
            denoise=bool(self.shading_config.get("denoise", True)),
            device=str(self.shading_config.get("device", "cuda")),
            proxy_reflectance=self.proxy_reflectance,
            overwrite=bool(self.shading_config.get("overwrite", False)),
        )

    def teardown(self) -> None:
        self.relighting_backend = None
        self.shading_backend = None

    @staticmethod
    def _view_dir(sample: PBREstimationSample2D) -> Path:
        view_dir = sample.rgb.parent.parent
        metadata_path = view_dir / "metadata.json"
        if not metadata_path.is_file():
            raise FileNotFoundError(f"Prepared view metadata not found: {metadata_path}")
        return view_dir

    def predict(
        self,
        samples: Sequence[PBREstimationSample2D],
        output_dir: Path,
    ) -> Mapping[str, Prediction2D]:
        if self.relighting_backend is None or self.shading_backend is None:
            raise RuntimeError("Call setup() before predict()")
        if not samples:
            return {}

        grouped: dict[tuple[str, str], list[PBREstimationSample2D]] = defaultdict(list)
        for sample in samples:
            grouped[(sample.object_id, sample.view_id)].append(sample)

        outputs: dict[str, Prediction2D] = {}
        # Keep caches beside predictions so evaluation does not mistake the
        # cache directory for a canonical prediction sample.
        cache_root = output_dir.parent / "oracle_cache"
        for (object_id, view_id), group in tqdm(
            sorted(grouped.items()),
            desc=f"PBROracle 2D [{len(grouped)} views]",
            unit="view",
        ):
            representative = group[0]
            view_dir = self._view_dir(representative)
            view_cache = cache_root / f"{object_id}__{view_id}"
            relightings = self.relighting_backend.generate(
                view_dir=view_dir,
                lights=self.lights,
                output_dir=view_cache / "relightings",
                overwrite=bool(self.relighting_config.get("overwrite", False)),
            )
            shading = self.shading_backend.render(
                metadata_path=view_dir / "metadata.json",
                lights=self.lights,
                output_dir=view_cache / "shading",
            )
            estimate = estimate_diffuse_albedo(
                relightings,
                shading,
                mask_path=representative.mask,
                proxy_reflectance=self.proxy_reflectance,
                energy_floor=self.energy_floor,
                clipping_start=self.clipping_start,
                minimum_clipping_weight=self.minimum_clipping_weight,
                max_ratio=self.max_ratio,
                log_epsilon=self.log_epsilon,
                normalized_dir=view_cache / "normalized",
            )

            Image.fromarray(estimate.albedo_srgb_u8, mode="RGB").save(
                view_cache / "albedo.png"
            )
            max_coverage = max(1, len(self.lights))
            coverage_u8 = np.rint(
                np.clip(estimate.coverage / max_coverage, 0.0, 1.0) * 255.0
            ).astype(np.uint8)
            Image.fromarray(coverage_u8, mode="L").save(view_cache / "coverage.png")
            confidence_u8 = np.rint(
                np.clip(estimate.confidence / max_coverage, 0.0, 1.0) * 255.0
            ).astype(np.uint8)
            Image.fromarray(confidence_u8, mode="L").save(
                view_cache / "confidence.png"
            )
            np.save(view_cache / "albedo_linear.npy", estimate.albedo_linear)

            metadata = {
                "method": "pbr_oracle",
                "quantity": "diffuse_albedo",
                "aggregation": "soft_weighted_log_space_channelwise_median",
                "lights": [light.to_dict() for light in self.lights],
                "proxy_reflectance": self.proxy_reflectance,
                "energy_floor": self.energy_floor,
                "clipping_start": self.clipping_start,
                "minimum_clipping_weight": self.minimum_clipping_weight,
                "roughness_status": "dummy",
                "roughness_value": self.dummy_roughness,
                "metallic_status": "dummy",
                "metallic_value": self.dummy_metallic,
                "limitations": [
                    "albedo is not PBR base color on metallic surfaces",
                    "roughness and metallic outputs are placeholders",
                ],
            }
            metadata_path = view_cache / "metadata.json"
            metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")

            mask = estimate.valid_mask
            roughness = np.zeros(mask.shape, dtype=np.uint8)
            metallic = np.zeros(mask.shape, dtype=np.uint8)
            roughness[mask] = int(round(self.dummy_roughness * 255.0))
            metallic[mask] = int(round(self.dummy_metallic * 255.0))

            for sample in group:
                outputs[sample.sample_id] = Prediction2D(
                    albedo=Image.fromarray(estimate.albedo_srgb_u8, mode="RGB"),
                    roughness=Image.fromarray(roughness, mode="L"),
                    metallic=Image.fromarray(metallic, mode="L"),
                ).save(save_dir=output_dir / sample.sample_id, mark_success=True)

        return outputs
