"""Native, baked-texture integration of Material Anything.

The upstream ``generate_texture_pbr_3d.py`` is a monolithic CLI pipeline. Its
ControlNet/RePaint stages are useful when an object has no appearance texture,
but are unnecessary for this benchmark: every evaluated object provides a
baked RGB texture. This adapter keeps the Material Anything material estimator,
progressive PBR-map conditioning, and UV refiner while running them directly
inside the benchmark process.

Each run is organized into two dataset-wide passes:
1. Pass A (Render, Geometry & Estimation): for every sample, render multi-view
   images, normals, similarity cache, and UV CCM maps (saved under
   sample_id/intermediate/stage1/), then run the progressive multi-view PBR
   estimator and bake per-view and coarse UV intermediates (saved under
   sample_id/intermediate/stage2/). The estimator is loaded into VRAM once
   for the whole pass and unloaded afterwards.
2. Pass B (UV Refinement & Hole Filling): load the refiner into VRAM once,
   then refine and Voronoi-fill each sample one by one (saved under
   sample_id/intermediate/stage3/), write the final prediction channels
   directly into the standard output layout, and unload the refiner.
"""

from __future__ import annotations

import gc
import logging
import shutil
import sys
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image
from tqdm.auto import tqdm

from src.data.pbr_estimation_dataset_3d import PBREstimationSample3D
from src.methods_3d.base import BaseMaterialEstimator3D, Prediction3D
from src.utils.glb import create_textured_glb

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class _UpstreamModules:
    """Typed container holding imported functions and constants from MaterialAnything."""

    cv2: Any
    kal: Any
    np: Any
    torch: Any
    TexturesUV: Any
    transforms: Any
    VIEWPOINTS: dict[str, Any]
    init_mesh_with_uv: Any
    init_renderer: Any
    init_flat_texel_shader: Any
    render_one_view: Any
    build_similarity_texture_cache_for_all_views: Any
    build_diffusion_mask: Any
    build_diffusion_materials: Any
    compose_quad_mask: Any
    backproject_from_image: Any
    bake_texture: Any
    MaterialEstimatorPipeline: Any
    UVRefinerPipeline: Any
    DDPMScheduler: Any
    voronoi_solve: Any


def _import_upstream(repo_root: Path) -> _UpstreamModules:
    """Import upstream Material Anything modules and heavy dependencies safely."""
    repo_path = str(repo_root)
    if repo_path not in sys.path:
        sys.path.insert(0, repo_path)

    import cv2
    import kaolin as kal
    import numpy as np
    import torch
    from pytorch3d.renderer import TexturesUV
    from torchvision import transforms

    try:
        from lib.constants import VIEWPOINTS
        from lib.mesh_helper import init_mesh_with_uv
        from lib.projection_helper import (
            backproject_from_image,
            bake_texture,
            build_diffusion_mask,
            build_diffusion_materials,
            build_similarity_texture_cache_for_all_views,
            compose_quad_mask,
            render_one_view,
        )
        from lib.render_helper import init_renderer
        from lib.shading_helper import init_flat_texel_shader
        from lib.voronoi import voronoi_solve
        from models.scheduling_ddpm import DDPMScheduler
        from pipelines.pipeline_stable_diffusion_switcher import (
            StableDiffusionPipeline as MaterialEstimatorPipeline,
        )
        from pipelines.pipeline_stable_diffusion_uv import (
            StableDiffusionPipeline as UVRefinerPipeline,
        )
    except ModuleNotFoundError as err:
        raise ModuleNotFoundError(
            f"Failed to import Material Anything components from {repo_root}. "
            "Ensure the submodule is present and the correct virtual environment is activated."
        ) from err

    return _UpstreamModules(
        cv2=cv2,
        kal=kal,
        np=np,
        torch=torch,
        TexturesUV=TexturesUV,
        transforms=transforms,
        VIEWPOINTS=VIEWPOINTS,
        init_mesh_with_uv=init_mesh_with_uv,
        init_renderer=init_renderer,
        init_flat_texel_shader=init_flat_texel_shader,
        render_one_view=render_one_view,
        build_similarity_texture_cache_for_all_views=build_similarity_texture_cache_for_all_views,
        build_diffusion_mask=build_diffusion_mask,
        build_diffusion_materials=build_diffusion_materials,
        compose_quad_mask=compose_quad_mask,
        backproject_from_image=backproject_from_image,
        bake_texture=bake_texture,
        MaterialEstimatorPipeline=MaterialEstimatorPipeline,
        UVRefinerPipeline=UVRefinerPipeline,
        DDPMScheduler=DDPMScheduler,
        voronoi_solve=voronoi_solve,
    )


@dataclass
class _CachedView:
    """Static render data for one fixed Material Anything camera."""

    distance: float
    elevation: float
    azimuth: float
    cameras: Any
    renderer: Any
    image: Image.Image
    normal: Image.Image


@dataclass
class _PreparedSample:
    """Stage-one data retained while stages two and three run."""

    mesh: Any
    faces: Any
    verts_uvs: Any
    views: list[_CachedView]
    similarity_cache: Any
    uv_ccm: Image.Image
    uv_mask: Image.Image


@dataclass
class _ProjectedMaterials:
    """Per-view predictions and masks ready for final UV fusion."""

    albedo_views: list[Image.Image]
    rm_views: list[Image.Image]
    bump_views: list[Image.Image]
    mask_views: list[Image.Image]


class MaterialAnythingEstimator3D(BaseMaterialEstimator3D):
    """Material Anything estimator with two-pass batch execution and VRAM offloading."""

    uv_correspondence = "identity"

    def __init__(
        self,
        *,
        name: str = "material_anything",
        project_root: str | Path = ".",
        repo_root: str | Path = "third_party/MaterialAnything",
        image2materials_model: str | Path = ".weights/material_anything/material_estimator",
        uvrefine_model: str | Path = ".weights/material_anything/material_refiner",
        viewpoint_preset: str = "objaverse",
        image_size: int = 768,
        uv_size: int = 1024,
        render_simple_factor: int = 4,
        view_threshold: float = 0.1,
        seed: int = 0,
        cleanup_intermediates: bool = True,
        save_intermediates: bool | None = None,
        # Kept only so existing Hydra overrides fail softly. Neither is used by
        # the upstream material-estimation helper for a baked-texture input.
        prompt: str | None = None,
        python_executable: str | Path | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(name=name, project_root=project_root, repo_root=repo_root)
        self.image2materials_model = self.resolve_model_path(image2materials_model)
        self.uvrefine_model = self.resolve_model_path(uvrefine_model)
        self.viewpoint_preset = viewpoint_preset
        self.image_size = int(image_size)
        self.uv_size = int(uv_size)
        self.render_simple_factor = int(render_simple_factor)
        self.view_threshold = float(view_threshold)
        self.seed = int(seed)
        if save_intermediates is not None:
            cleanup_intermediates = not save_intermediates
        self.cleanup_intermediates = bool(cleanup_intermediates)
        self.prompt = prompt
        self.python_executable = python_executable

        self._device: Any | None = None
        self._up: _UpstreamModules | None = None
        self._material_model: Any | None = None
        self._uv_refiner: Any | None = None

    def resolve_model_path(self, path: str | Path) -> Path:
        path = Path(path)
        if path.is_absolute():
            return path.resolve()
        project_path = (self.project_root / path).resolve()
        return project_path if project_path.exists() else (self.repo_root / path).resolve()

    def setup(self) -> None:
        """Validate assets and import the upstream internals in this process."""
        super().setup()
        if not self.image2materials_model.is_dir():
            raise FileNotFoundError(
                "Material Anything estimator model not found at "
                f"{self.image2materials_model}. Run "
                "third_party/MaterialAnything/download_models.sh."
            )
        if not self.uvrefine_model.is_dir():
            raise FileNotFoundError(
                "Material Anything refiner model not found at "
                f"{self.uvrefine_model}. Run "
                "third_party/MaterialAnything/download_models.sh."
            )

        up = _import_upstream(self.repo_root)

        if not up.torch.cuda.is_available():
            raise RuntimeError("Material Anything requires a CUDA-capable PyTorch3D environment.")

        if self.viewpoint_preset not in up.VIEWPOINTS:
            available = ", ".join(str(key) for key in up.VIEWPOINTS)
            raise ValueError(
                f"Unknown Material Anything viewpoint preset {self.viewpoint_preset!r}; "
                f"available presets: {available}."
            )

        self._device = up.torch.device("cuda:0")
        self._up = up

        if self.prompt:
            logger.info(
                "Ignoring Material Anything text prompt for baked-texture inference; "
                "the upstream material-estimation helper uses an empty prompt."
            )

    def teardown(self) -> None:
        self._unload_estimator_model()
        self._unload_refiner_model()
        self._up = None
        self._device = None

    # #########################################################################
    # ### VENDORED CODE START
    # Source: third_party/MaterialAnything (MIT License, (c) 2024 3D Topia)
    # Adapted in this file so the benchmark can run the estimator and refiner
    # in-process, skipping the ControlNet/RePaint stages of the upstream CLI
    # script. Origin of each method:
    #   - _load_estimator_model/_load_refiner_model: lib/diffusion_helper.py
    #     get_image2materials/get_uvrefiner (lazy load added; _unload_* are new)
    #   - _run_material_estimator: lib/diffusion_helper.py apply_material_estimation
    #   - _run_uv_refiner: lib/diffusion_helper.py apply_uv_refinement
    #   - _set_baked_texture: scripts/generate_texture_pbr_3d.py texture setup
    #   - _build_uv_ccm: scripts/generate_texture_pbr_3d.py uv_to_3d + dilation
    #   - _camera_space_normal: lib/projection_helper.py
    #     render_one_view_and_build_masks_materials (normal-map conversion)
    #   - _estimate_and_project: scripts/generate_texture_pbr_3d.py main
    #     multi-view generation loop
    #   - _bake_coarse_textures: scripts/generate_texture_pbr_3d.py final bake
    #   - _refine_and_fuse/_fill_uv_holes: scripts/generate_texture_pbr_3d.py
    #     final UV refinement and Voronoi fill
    # Modifications vs upstream: benchmark baked texture instead of
    # texture_kd.png, configurable generator seed instead of a fixed seed of 0,
    # intermediates saved under the benchmark output layout, and no
    # ControlNet/RePaint update stage. NOTE _bake_coarse_textures passes the
    # raw similarity cache to bake_texture; upstream first masks it with the
    # projected per-view masks (generate_texture_pbr_3d.py lines 681-684).
    # #########################################################################

    def _load_estimator_model(self) -> None:
        if self._material_model is not None:
            return
        if self._up is None:
            raise RuntimeError("Call setup() before _load_estimator_model().")
        logger.info("Loading Material Anything estimator into GPU VRAM.")
        up = self._up
        torch = up.torch
        self._material_model = up.MaterialEstimatorPipeline.from_pretrained(
            str(self.image2materials_model), torch_dtype=torch.float16
        ).to(self._device)
        self._material_model.scheduler = up.DDPMScheduler.from_pretrained(
            str(self.image2materials_model), subfolder="scheduler"
        )

    def _unload_estimator_model(self) -> None:
        if self._material_model is not None:
            logger.info("Unloading Material Anything estimator from GPU VRAM.")
            self._material_model = None
            if self._up is not None:
                self._up.torch.cuda.empty_cache()

    def _load_refiner_model(self) -> None:
        if self._uv_refiner is not None:
            return
        if self._up is None:
            raise RuntimeError("Call setup() before _load_refiner_model().")
        logger.info("Loading Material Anything UV refiner into GPU VRAM.")
        up = self._up
        torch = up.torch
        self._uv_refiner = up.UVRefinerPipeline.from_pretrained(
            str(self.uvrefine_model), torch_dtype=torch.float16
        ).to(self._device)

    def _unload_refiner_model(self) -> None:
        if self._uv_refiner is not None:
            logger.info("Unloading Material Anything UV refiner from GPU VRAM.")
            self._uv_refiner = None
            if self._up is not None:
                self._up.torch.cuda.empty_cache()

    def _run_material_estimator(
        self, image: Image.Image, normal: Image.Image, materials: Any, mask: Any
    ) -> tuple[Image.Image, Image.Image, Image.Image]:
        """Run the upstream estimator helper without importing ControlNet code."""
        if self._up is None or self._material_model is None:
            raise RuntimeError("Setup and estimator model initialization required.")
        generator = self._up.torch.Generator("cuda").manual_seed(self.seed)
        albedo, rm, bump = self._material_model(
            prompt=[""],
            cond_image=[image],
            normal_image=[normal],
            init_materials=materials,
            masks=mask,
            num_inference_steps=50,
            guidance_scale=1.0,
            generator=generator,
            height=self.image_size,
            width=self.image_size,
        ).images
        return tuple(img.resize((self.image_size, self.image_size)) for img in (albedo, rm, bump))

    def _run_uv_refiner(
        self, albedo: Image.Image, rm: Image.Image, bump: Image.Image, ccm: Image.Image
    ) -> tuple[Image.Image, Image.Image, Image.Image]:
        """Run the upstream UV helper; its mask argument is unused upstream."""
        if self._up is None or self._uv_refiner is None:
            raise RuntimeError("Setup and UV refiner model initialization required.")
        generator = self._up.torch.Generator("cuda").manual_seed(self.seed)
        return tuple(
            self._uv_refiner(
                prompt="",
                ccm_image=[ccm],
                albedo_image=[albedo],
                rm_image=[rm],
                bump_image=[bump],
                num_inference_steps=50,
                guidance_scale=1.0,
                generator=generator,
                height=self.uv_size,
                width=self.uv_size,
            ).images
        )

    def _set_baked_texture(
        self, mesh: Any, faces: Any, verts_uvs: Any, baked_texture: Any | None
    ) -> Any:
        """Attach the benchmark RGB texture to the normalized upstream mesh."""
        up = self._up
        if up is None:
            raise RuntimeError("Call setup() first.")
        if isinstance(baked_texture, up.torch.Tensor):
            baked_np = (
                baked_texture.permute(1, 2, 0).cpu().numpy() * 255.0
            ).clip(0, 255).astype("uint8")
            texture = Image.fromarray(baked_np).convert("RGB")
        elif baked_texture is not None and isinstance(baked_texture, (str, Path)) and Path(baked_texture).is_file():
            texture = Image.open(baked_texture).convert("RGB")
        else:
            logger.warning("No texture provided; using a white base-color texture.")
            texture = Image.new("RGB", (self.uv_size, self.uv_size), (255, 255, 255))
        texture = texture.resize((self.uv_size, self.uv_size), Image.Resampling.LANCZOS)
        if verts_uvs is None or faces.textures_idx is None:
            raise ValueError(
                "Material Anything requires an OBJ with UV coordinates and face UV indices."
            )
        mesh.textures = up.TexturesUV(
            maps=up.transforms.ToTensor()(texture)[None, ...]
            .permute(0, 2, 3, 1)
            .to(self._device),
            faces_uvs=faces.textures_idx[None, ...],
            verts_uvs=verts_uvs[None, ...],
        )
        return verts_uvs

    def _build_uv_ccm(self, mesh: Any) -> tuple[Image.Image, Image.Image]:
        """Build the upstream canonical-coordinate map in UV space."""
        up = self._up
        if up is None:
            raise RuntimeError("Call setup() first.")
        torch, np, cv2, kal = up.torch, up.np, up.cv2, up.kal
        textures = mesh.textures
        verts_uvs = textures.verts_uvs_padded()[0]
        faces_uvs = textures.faces_uvs_padded()[0]
        verts = mesh.verts_padded()[0]
        faces = mesh.faces_padded()[0]
        uv_face_attr = verts_uvs[faces_uvs].unsqueeze(0)
        face_vertices_world = kal.ops.mesh.index_vertices_by_faces(verts.unsqueeze(0), faces)
        face_vertices_z = torch.zeros_like(face_vertices_world[..., -1], device=self._device)
        uv_position, face_idx = kal.render.mesh.rasterize(
            self.uv_size, self.uv_size, face_vertices_z, uv_face_attr * 2 - 1,
            face_features=face_vertices_world,
        )
        uv_position = torch.clamp(uv_position, -1, 1)
        uv_position = uv_position * 2 / 2 + 0.5
        uv_position[face_idx == -1] = 1
        uv_array = (uv_position[0].cpu().numpy() * 255).astype(np.uint8)

        # Match the upstream dilation so the refiner sees geometry at UV seams.
        mask = np.any(uv_array != [255, 255, 255], axis=-1)[:, :, None].repeat(3, axis=2).astype(np.uint8) * 255
        kernel = np.ones((3, 3), np.uint8)
        dilated_mask = cv2.dilate(mask, kernel, iterations=5)
        masked = uv_array.copy()
        masked[np.all(masked == [255, 255, 255], axis=-1)] = 0
        dilated = cv2.dilate(masked, kernel, iterations=5)
        ccm = Image.fromarray(np.where(dilated_mask == 255, dilated, uv_array).astype(np.uint8))
        uv_mask = Image.fromarray(
            np.any(np.array(ccm) != [255, 255, 255], axis=-1).astype(np.uint8) * 255
        )
        return ccm, uv_mask

    def _camera_space_normal(self, normals: Any, cameras: Any) -> Image.Image:
        """Use the same world-to-camera normal conversion as the upstream script."""
        up = self._up
        if up is None:
            raise RuntimeError("Call setup() first.")
        normal = normals[0].cpu().permute(2, 0, 1)
        valid = (abs(normals[0]).sum(-1) > 0).float().cpu()
        transform = up.torch.tensor(
            [[-1, 0, 0], [0, 1, 0], [0, 0, -1]], dtype=normal.dtype
        )
        normal = normal.permute(1, 2, 0).reshape(-1, 3)
        normal = (cameras.R[0].cpu().T @ normal.T).T @ transform
        normal = normal.reshape(self.image_size, self.image_size, 3).permute(2, 0, 1)
        normal = (normal + 1.0) / 2.0
        normal = normal * valid + (1 - valid)
        return up.transforms.ToPILImage()(normal).convert("RGB")

    def _estimate_and_project(
        self, prepared: _PreparedSample, stage2_dir: Path, sample_id: str
    ) -> _ProjectedMaterials:
        """Pass 2: estimation loop and per-view PBR projection in stage2/."""
        up = self._up
        if up is None:
            raise RuntimeError("Call setup() first.")
        torch, np = up.torch, up.np

        with torch.no_grad():
            white = Image.new("RGB", (self.uv_size, self.uv_size), (255, 255, 255))
            albedo_uv, rm_uv, bump_uv = white.copy(), white.copy(), white.copy()
            albedo_coverage = torch.zeros((self.uv_size, self.uv_size), device=self._device)
            rm_coverage = torch.zeros_like(albedo_coverage)
            bump_coverage = torch.zeros_like(albedo_coverage)
            albedo_views: list[Image.Image] = []
            rm_views: list[Image.Image] = []
            bump_views: list[Image.Image] = []
            mask_views: list[Image.Image] = []

            view_iter = tqdm(
                enumerate(prepared.views),
                total=len(prepared.views),
                desc=f"Estimating 3D views [{sample_id}]",
                leave=False,
            )
            for index, view in view_iter:
                # Re-create flat_renderer on demand for this view to save memory
                flat_renderer = up.init_renderer(
                    view.cameras,
                    shader=up.init_flat_texel_shader(camera=view.cameras, device=self._device),
                    image_size=self.image_size,
                    faces_per_pixel=1,
                )
                new_mask, update_mask, old_mask, _ = up.build_diffusion_mask(
                    (prepared.mesh, prepared.faces, prepared.verts_uvs), flat_renderer,
                    albedo_coverage, prepared.similarity_cache, index, self._device,
                    self.image_size, view_threshold=self.view_threshold,
                )
                _, _, new_mask_tensor, all_mask_tensor, _ = up.compose_quad_mask(
                    new_mask, update_mask, old_mask, self._device
                )
                materials = up.build_diffusion_materials(
                    (prepared.mesh, prepared.faces, prepared.verts_uvs), view.renderer,
                    albedo_uv, rm_uv, bump_uv, self._device,
                )
                albedo, rm, bump = self._run_material_estimator(
                    view.image, view.normal, materials, 1 - new_mask_tensor
                )
                albedo_views.append(albedo)
                rm_views.append(rm)
                bump_views.append(bump)

                stage2_dir.mkdir(parents=True, exist_ok=True)
                albedo.save(stage2_dir / f"view_{index:02d}_albedo.png")
                rm.save(stage2_dir / f"view_{index:02d}_rm.png")
                bump.save(stage2_dir / f"view_{index:02d}_bump.png")

                albedo_uv, _, albedo_coverage = up.backproject_from_image(
                    prepared.mesh, prepared.faces, prepared.verts_uvs, view.cameras,
                    albedo, new_mask, update_mask, albedo_uv, albedo_coverage,
                    self.image_size * self.render_simple_factor, self.uv_size, 1,
                    self._device, blending_weights=0.0,
                )
                rm_uv, _, rm_coverage = up.backproject_from_image(
                    prepared.mesh, prepared.faces, prepared.verts_uvs, view.cameras,
                    rm, new_mask, update_mask, rm_uv, rm_coverage,
                    self.image_size * self.render_simple_factor, self.uv_size, 1,
                    self._device, blending_weights=0.0,
                )
                bump_uv, _, bump_coverage = up.backproject_from_image(
                    prepared.mesh, prepared.faces, prepared.verts_uvs, view.cameras,
                    bump, new_mask, update_mask, bump_uv, bump_coverage,
                    self.image_size * self.render_simple_factor, self.uv_size, 1,
                    self._device, blending_weights=0.0,
                )
                all_mask = Image.fromarray(
                    (all_mask_tensor[0].cpu().numpy() * 255).astype(np.uint8)
                )
                empty_mask = Image.new("RGB", (self.uv_size, self.uv_size), (0, 0, 0))
                view_mask, _, _ = up.backproject_from_image(
                    prepared.mesh, prepared.faces, prepared.verts_uvs, view.cameras,
                    all_mask.convert("RGB"), all_mask, all_mask, empty_mask,
                    torch.zeros_like(albedo_coverage),
                    self.image_size * self.render_simple_factor, self.uv_size, 1,
                    self._device, blending_weights=1.0,
                )
                stage2_dir.mkdir(parents=True, exist_ok=True)
                view_mask.save(stage2_dir / f"view_{index:02d}_mask.png")
                mask_views.append(view_mask)

            return _ProjectedMaterials(albedo_views, rm_views, bump_views, mask_views)

    def _bake_coarse_textures(
        self,
        prepared: _PreparedSample,
        projected: _ProjectedMaterials,
        stage2_dir: Path,
    ) -> None:
        """Bake multi-view predictions into coarse UV space and save to stage2/."""
        up = self._up
        if up is None:
            raise RuntimeError("Call setup() first.")

        distances = [view.distance for view in prepared.views]
        elevations = [view.elevation for view in prepared.views]
        azimuths = [view.azimuth for view in prepared.views]
        weights = prepared.similarity_cache

        with up.torch.enable_grad():
            albedo_coarse = up.bake_texture(
                projected.albedo_views, prepared.mesh, distances, elevations,
                azimuths, weights, self.image_size, self.uv_size, device=self._device,
            )
            rm_coarse = up.bake_texture(
                projected.rm_views, prepared.mesh, distances, elevations,
                azimuths, weights, self.image_size, self.uv_size, device=self._device,
            )
            bump_coarse = up.bake_texture(
                projected.bump_views, prepared.mesh, distances, elevations,
                azimuths, weights, self.image_size, self.uv_size, exp=6,
                device=self._device,
            )

        stage2_dir.mkdir(parents=True, exist_ok=True)
        albedo_coarse.save(stage2_dir / "coarse_albedo_uv.png")
        rm_coarse.save(stage2_dir / "coarse_rm_uv.png")
        bump_coarse.save(stage2_dir / "coarse_bump_uv.png")

    def _refine_and_fuse(
        self,
        stage1_dir: Path,
        stage2_dir: Path,
        stage3_dir: Path,
    ) -> tuple[Image.Image, Image.Image, Image.Image]:
        """Pass 3: load coarse maps and geometry from disk, refine, and fill holes."""
        up = self._up
        if up is None:
            raise RuntimeError("Call setup() first.")

        with up.torch.no_grad():
            albedo_coarse = Image.open(stage2_dir / "coarse_albedo_uv.png").convert("RGB")
            rm_coarse = Image.open(stage2_dir / "coarse_rm_uv.png").convert("RGB")
            bump_coarse = Image.open(stage2_dir / "coarse_bump_uv.png").convert("RGB")
            uv_ccm = Image.open(stage1_dir / "uv_ccm.png").convert("RGB")
            uv_mask = Image.open(stage1_dir / "uv_mask.png").convert("L")

            albedo_ref, rm_ref, bump_ref = self._run_uv_refiner(
                albedo_coarse, rm_coarse, bump_coarse, uv_ccm
            )
            stage3_dir.mkdir(parents=True, exist_ok=True)
            albedo_ref.save(stage3_dir / "refined_albedo_uv.png")
            rm_ref.save(stage3_dir / "refined_rm_uv.png")
            bump_ref.save(stage3_dir / "refined_bump_uv.png")

            albedo_final = self._fill_uv_holes(albedo_ref, uv_mask)
            rm_final = self._fill_uv_holes(rm_ref, uv_mask)
            bump_final = self._fill_uv_holes(bump_ref, uv_mask)

            albedo_final.save(stage3_dir / "final_albedo_uv.png")
            rm_final.save(stage3_dir / "final_rm_uv.png")
            bump_final.save(stage3_dir / "final_bump_uv.png")

            return albedo_final, rm_final, bump_final

    def _fill_uv_holes(self, image: Image.Image, uv_mask: Image.Image) -> Image.Image:
        up = self._up
        if up is None:
            raise RuntimeError("Call setup() first.")
        tensor = up.transforms.ToTensor()(image).to(self._device)
        mask = up.transforms.ToTensor()(uv_mask).to(self._device)[0]
        filled = up.voronoi_solve(tensor.permute(1, 2, 0), mask, self._device)
        return Image.fromarray((filled.cpu().numpy() * 255).astype(up.np.uint8))

    # ### VENDORED CODE END
    # #########################################################################

    def _prepare_sample(
        self, sample: PBREstimationSample3D, stage1_dir: Path
    ) -> _PreparedSample:
        """Pass 1: create static baked-RGB camera views and save geometry intermediates in stage1/."""
        up = self._up
        if up is None:
            raise RuntimeError("Call setup() first.")

        with up.torch.no_grad():
            mesh, _, faces, aux, _, _, _ = up.init_mesh_with_uv(
                str(sample.mesh_path), str(sample.mesh_path), self._device
            )
            verts_uvs = self._set_baked_texture(
                mesh, faces, aux.verts_uvs, sample.baked_texture
            )
            viewpoints = up.VIEWPOINTS[self.viewpoint_preset]
            elevations = list(viewpoints["elev"])
            azimuths = list(viewpoints["azim"])
            distances = [1.0] * len(elevations)
            cached_views: list[_CachedView] = []
            for idx, (distance, elevation, azimuth) in enumerate(
                zip(distances, elevations, azimuths, strict=True)
            ):
                cameras, renderer, images, normals, _, _, _ = up.render_one_view(
                    mesh, distance, elevation, azimuth, self.image_size, 1, self._device
                )
                image = up.transforms.ToPILImage()(images[0].cpu().permute(2, 0, 1)).convert("RGB")
                normal_img = self._camera_space_normal(normals, cameras)
                cached_views.append(
                    _CachedView(
                        distance, elevation, azimuth, cameras, renderer, image, normal_img
                    )
                )
                stage1_dir.mkdir(parents=True, exist_ok=True)
                image.save(stage1_dir / f"view_{idx:02d}_image.png")
                normal_img.save(stage1_dir / f"view_{idx:02d}_normal.png")

            similarity_cache = up.build_similarity_texture_cache_for_all_views(
                mesh, faces, verts_uvs, distances, elevations, azimuths,
                self.image_size, self.image_size * self.render_simple_factor,
                self.uv_size, 1, self._device,
            )
            ccm, uv_mask = self._build_uv_ccm(mesh)
            stage1_dir.mkdir(parents=True, exist_ok=True)
            ccm.save(stage1_dir / "uv_ccm.png")
            uv_mask.save(stage1_dir / "uv_mask.png")

            return _PreparedSample(mesh, faces, verts_uvs, cached_views, similarity_cache, ccm, uv_mask)

    def predict_over_dataset(
        self,
        samples: Sequence[PBREstimationSample3D],
        output_dir: str | Path,
    ) -> Iterator[Prediction3D]:
        """Predict 3D material maps for a collection of samples."""
        if self._up is None or self._device is None:
            raise RuntimeError("Call setup() before predict_over_dataset().")

        output_path = Path(output_dir)

        # Pass A: material estimation & texture baking for all samples
        logger.info(
            "Material Anything Pass A/2: estimation & baking for %d samples...",
            len(samples),
        )
        self._load_estimator_model()
        try:
            for sample in tqdm(
                samples, desc="Pass A/2 (Estimation & Baking)", leave=True
            ):
                sample_dir = output_path / sample.sample_id
                stage1_dir = sample_dir / "intermediate" / "stage1"
                stage2_dir = sample_dir / "intermediate" / "stage2"

                prepared = self._prepare_sample(sample, stage1_dir)
                projected = self._estimate_and_project(
                    prepared, stage2_dir, sample_id=sample.sample_id
                )
                self._bake_coarse_textures(prepared, projected, stage2_dir)

                del prepared, projected
                gc.collect()
                if self._up.torch.cuda.is_available():
                    self._up.torch.cuda.empty_cache()
        finally:
            self._unload_estimator_model()

        # Pass B: UV refinement & hole filling for all samples
        logger.info(
            "Material Anything Pass B/2: UV refinement & hole filling for %d samples...",
            len(samples),
        )
        self._load_refiner_model()
        try:
            for sample in tqdm(
                samples,
                desc="Pass B/2 (Refinement & Hole Filling)",
                leave=True,
            ):
                sample_dir = output_path / sample.sample_id
                stage1_dir = sample_dir / "intermediate" / "stage1"
                stage2_dir = sample_dir / "intermediate" / "stage2"
                stage3_dir = sample_dir / "intermediate" / "stage3"

                albedo, rm, _ = self._refine_and_fuse(
                    stage1_dir, stage2_dir, stage3_dir
                )
                _, roughness, metallic = rm.convert("RGB").split()

                albedo_path = sample_dir / "albedo.png"
                roughness_path = sample_dir / "roughness.png"
                metallic_path = sample_dir / "metallic.png"

                albedo.save(albedo_path)
                roughness.save(roughness_path)
                metallic.save(metallic_path)
                mesh_path = create_textured_glb(
                    sample.mesh_path,
                    {
                        "albedo": albedo,
                        "roughness": roughness,
                        "metallic": metallic,
                    },
                    sample_dir / "mesh.glb",
                )

                artifacts: dict[str, Path] = {}
                if not self.cleanup_intermediates:
                    artifacts = {
                        "ccm": stage1_dir / "uv_ccm.png",
                        "coarse_albedo": stage2_dir / "coarse_albedo_uv.png",
                        "coarse_rm": stage2_dir / "coarse_rm_uv.png",
                        "coarse_bump": stage2_dir / "coarse_bump_uv.png",
                        "refined_albedo": stage3_dir / "refined_albedo_uv.png",
                        "refined_rm": stage3_dir / "refined_rm_uv.png",
                        "refined_bump": stage3_dir / "refined_bump_uv.png",
                    }

                yield Prediction3D(
                    sample_id=sample.sample_id,
                    pbr_asset_glb=mesh_path,
                    artifacts=artifacts,
                )

                if self.cleanup_intermediates:
                    shutil.rmtree(sample_dir / "intermediate", ignore_errors=True)
        finally:
            self._unload_refiner_model()

