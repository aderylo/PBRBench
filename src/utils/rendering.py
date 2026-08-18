"""Shared 2D reference-image rendering utilities for 3D methods."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
import torch

from src.data.pbr_estimation_dataset_3d import PBREstimationSample3D


def render_reference_image(
    sample: PBREstimationSample3D,
    *,
    camera_distance: float = 1.6,
    camera_elevation: float = 0.0,
    camera_azimuth: float = 0.0,
    camera_focal_length: float = 1.375,
    resolution: int = 512,
    device: str = "cuda",
    output_path: str | Path | None = None,
) -> Image.Image:
    """Render the mesh textured with its baked UV atlas into a canonical 2D reference view.

    Renders the sample's mesh with the ground truth baked texture using PyTorch3D and
    returns the composited-on-white RGB image. If ``output_path`` is given, the RGBA
    render is also persisted there.

    Args:
        sample: The 3D sample whose mesh and baked texture should be rendered.
        camera_distance: Distance from the camera to the mesh origin.
        camera_elevation: Camera elevation angle in degrees.
        camera_azimuth: Camera azimuth angle in degrees.
        camera_focal_length: Camera focal length (in pixel units when ``image_size`` is
            given in pixels).
        resolution: Side length (pixels) of the square rendered image.
        device: Torch device used for rendering.
        output_path: Optional location to persist the RGBA render (``reference_image.png``).
    """
    from pytorch3d.io import load_obj, load_objs_as_meshes
    from pytorch3d.renderer import (
        AmbientLights,
        BlendParams,
        MeshRasterizer,
        MeshRendererWithFragments,
        PerspectiveCameras,
        RasterizationSettings,
        SoftPhongShader,
        TexturesUV,
        look_at_view_transform,
    )

    device = torch.device(device)

    with torch.no_grad():
        _, faces, aux = load_obj(str(sample.mesh_path), device=device)
        p3d_mesh = load_objs_as_meshes([str(sample.mesh_path)], device=device)

        baked_texture: Any = sample.baked_texture
        if isinstance(baked_texture, torch.Tensor):
            tex_tensor = baked_texture.permute(1, 2, 0).to(device)
        elif (
            baked_texture is not None
            and isinstance(baked_texture, (str, Path))
            and Path(baked_texture).is_file()
        ):
            tex_img = Image.open(baked_texture).convert("RGB")
            tex_tensor = torch.from_numpy(
                np.asarray(tex_img, dtype=np.float32) / 255.0
            ).to(device)
        else:
            tex_tensor = torch.ones((1024, 1024, 3), device=device)

        p3d_mesh.textures = TexturesUV(
            maps=tex_tensor[None, ...],
            faces_uvs=faces.textures_idx[None, ...],
            verts_uvs=aux.verts_uvs[None, ...],
        )

        R, T = look_at_view_transform(
            dist=camera_distance,
            elev=camera_elevation,
            azim=camera_azimuth,
        )
        img_size = torch.tensor([[resolution, resolution]], device=device)
        cameras = PerspectiveCameras(
            focal_length=camera_focal_length,
            R=R,
            T=T,
            device=device,
            image_size=img_size,
        )
        raster_settings = RasterizationSettings(
            image_size=resolution,
            faces_per_pixel=1,
        )
        lights = AmbientLights(device=device)
        shader = SoftPhongShader(
            cameras=cameras,
            lights=lights,
            device=device,
            blend_params=BlendParams(background_color=(1.0, 1.0, 1.0)),
        )
        renderer = MeshRendererWithFragments(
            rasterizer=MeshRasterizer(cameras=cameras, raster_settings=raster_settings),
            shader=shader,
        )
        images, fragments = renderer(p3d_mesh)

        rgb = images[0, ..., :3].clamp(0, 1)
        alpha = (fragments.pix_to_face[0, ..., 0] >= 0).float()
        rgba = torch.cat([rgb, alpha[..., None]], dim=-1)
        rgba_uint8 = (rgba.cpu().numpy() * 255.0).round().astype(np.uint8)

        if output_path is not None:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(rgba_uint8, "RGBA").save(output_path)

        # Composite onto white background for image conditioning
        rgb_composed = rgb * alpha[..., None] + 1.0 * (1.0 - alpha[..., None])
        return Image.fromarray(
            (rgb_composed.cpu().numpy() * 255.0).round().astype(np.uint8), "RGB"
        )