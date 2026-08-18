"""Small GLB helpers used by 3D material-estimation postprocessing."""

from __future__ import annotations

from collections.abc import Collection, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
from PIL import Image
from scipy.spatial import cKDTree


PBRChannel = Literal["albedo", "roughness", "metallic"]


@dataclass(frozen=True)
class UVTextureTransfer:
    """PBR textures represented in the target mesh's UV layout."""

    textures: dict[PBRChannel, Image.Image]
    uv_mask: Image.Image
    correspondence: str
    valid_fraction: float


def _load_mesh(path: str | Path):
    """Load one textured mesh without silently changing its indexing."""
    import trimesh

    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    mesh = trimesh.load(path, force="mesh", process=False)
    if not isinstance(mesh, trimesh.Trimesh):
        raise ValueError(f"Expected one mesh in {path}, got {type(mesh).__name__}")
    return mesh


def _factor(material, name: str, length: int, default: float) -> np.ndarray:
    value = getattr(material, name, None)
    if value is None:
        return np.full(length, default, dtype=np.float32)
    array = np.asarray(value, dtype=np.float32).reshape(-1)
    if array.size < length:
        raise ValueError(f"Invalid glTF material factor {name}: {value!r}")
    array = array[:length]
    if array.max(initial=0.0) > 1.0:
        array /= 255.0
    return array


def _image_array(image, mode: str) -> np.ndarray | None:
    if image is None:
        return None
    if not isinstance(image, Image.Image):
        image = Image.fromarray(np.asarray(image))
    return np.asarray(image.convert(mode), dtype=np.float32) / 255.0


def extract_pbr_textures(
    textured_asset: str | Path,
    *,
    channels: Collection[PBRChannel] = ("albedo", "roughness", "metallic"),
) -> dict[PBRChannel, Image.Image]:
    """Extract effective base-color, roughness, and metallic images from a GLB.

    This intentionally supports the single-mesh, single-material metallic-
    roughness assets produced by the current adapters. Material factors are
    folded into the returned images.
    """
    requested = set(channels)
    unknown = requested - {"albedo", "roughness", "metallic"}
    if unknown:
        raise ValueError(f"Unknown PBR channels: {sorted(unknown)}")

    mesh = _load_mesh(textured_asset)
    visual = mesh.visual
    uv = getattr(visual, "uv", None)
    material = getattr(visual, "material", None)
    if uv is None or material is None:
        raise ValueError(f"Asset has no UV-mapped material: {textured_asset}")

    base = _image_array(getattr(material, "baseColorTexture", None), "RGBA")
    mr = _image_array(getattr(material, "metallicRoughnessTexture", None), "RGB")
    size = (
        (base.shape[1], base.shape[0])
        if base is not None
        else ((mr.shape[1], mr.shape[0]) if mr is not None else (1, 1))
    )

    result: dict[PBRChannel, Image.Image] = {}
    if "albedo" in requested:
        factor = _factor(material, "baseColorFactor", 4, 1.0)
        if base is None:
            base = np.ones((size[1], size[0], 4), dtype=np.float32)
        albedo = np.clip(base[..., :3] * factor[:3], 0.0, 1.0)
        result["albedo"] = Image.fromarray(
            np.round(albedo * 255.0).astype(np.uint8), "RGB"
        )

    roughness_factor = float(_factor(material, "roughnessFactor", 1, 1.0)[0])
    metallic_factor = float(_factor(material, "metallicFactor", 1, 1.0)[0])
    if mr is None:
        mr = np.ones((size[1], size[0], 3), dtype=np.float32)
    if "roughness" in requested:
        roughness = np.clip(mr[..., 1] * roughness_factor, 0.0, 1.0)
        result["roughness"] = Image.fromarray(
            np.round(roughness * 255.0).astype(np.uint8), "L"
        )
    if "metallic" in requested:
        metallic = np.clip(mr[..., 2] * metallic_factor, 0.0, 1.0)
        result["metallic"] = Image.fromarray(
            np.round(metallic * 255.0).astype(np.uint8), "L"
        )
    return result


def _sample_bilinear(image: Image.Image, uv: np.ndarray) -> np.ndarray:
    """Sample a PIL image at glTF UV coordinates using repeat wrapping."""
    array = np.asarray(image, dtype=np.float32) / 255.0
    if array.ndim == 2:
        array = array[..., None]
    height, width = array.shape[:2]
    uv = np.mod(uv, 1.0)
    x = uv[:, 0] * width - 0.5
    y = (1.0 - uv[:, 1]) * height - 0.5
    x0 = np.floor(x).astype(np.int64)
    y0 = np.floor(y).astype(np.int64)
    fx = (x - x0)[:, None]
    fy = (y - y0)[:, None]
    x0 %= width
    y0 %= height
    x1 = (x0 + 1) % width
    y1 = (y0 + 1) % height
    return (
        array[y0, x0] * (1.0 - fx) * (1.0 - fy)
        + array[y0, x1] * fx * (1.0 - fy)
        + array[y1, x0] * (1.0 - fx) * fy
        + array[y1, x1] * fx * fy
    )


def remap_uv_textures(
    source_asset: str | Path,
    target_mesh: str | Path,
    *,
    resolution: int,
    correspondence: Literal["identity", "topology", "auto"] = "auto",
) -> UVTextureTransfer:
    """Resample a GLB material into the UV layout of matching target geometry.

    Version one requires corresponding face order. It supports uniform object
    normalization and duplicated vertices introduced at new UV seams, but it
    does not guess a closest-surface correspondence.
    """
    source = _load_mesh(source_asset)
    target = _load_mesh(target_mesh)
    if correspondence not in {"identity", "topology", "auto"}:
        raise ValueError(f"Unknown correspondence mode: {correspondence}")
    source_uv = getattr(source.visual, "uv", None)
    target_uv = getattr(target.visual, "uv", None)
    if source_uv is None or target_uv is None:
        raise ValueError("Both source and target meshes must have UV coordinates")
    source_vertices = np.asarray(source.vertices)
    target_vertices = np.asarray(target.vertices)
    source_extent = np.ptp(source_vertices, axis=0).max()
    target_extent = np.ptp(target_vertices, axis=0).max()
    if source_extent <= 0 or target_extent <= 0:
        raise ValueError("Source or target mesh has zero extent")
    source_normalized = (
        source_vertices - (source.bounds[0] + source.bounds[1]) / 2.0
    ) / source_extent
    target_normalized = (
        target_vertices - (target.bounds[0] + target.bounds[1]) / 2.0
    ) / target_extent
    if source.faces.shape != target.faces.shape:
        raise ValueError("Source and target face counts do not correspond")

    source_triangles = source_normalized[source.faces]
    target_triangles = target_normalized[target.faces]
    face_distances, source_face_for_target = cKDTree(
        source_triangles.mean(axis=1)
    ).query(target_triangles.mean(axis=1), k=1)
    if float(face_distances.max()) > 1e-5:
        raise ValueError("Source and target triangle positions do not correspond")
    if len(np.unique(source_face_for_target)) != len(source.faces):
        raise ValueError("Triangle correspondence is not one-to-one")
    source_uv_triangles = np.empty((len(target.faces), 3, 2), dtype=np.float64)
    for target_face_index, target_face in enumerate(target.faces):
        source_face_index = source_face_for_target[target_face_index]
        source_face = source.faces[source_face_index]
        corner_distances = np.linalg.norm(
            target_triangles[target_face_index, :, None, :]
            - source_triangles[source_face_index, None, :, :],
            axis=-1,
        )
        source_corner_for_target = corner_distances.argmin(axis=1)
        if (
            len(np.unique(source_corner_for_target)) != 3
            or corner_distances[
                np.arange(3), source_corner_for_target
            ].max() > 1e-5
        ):
            raise ValueError("Could not resolve source triangle corner order")
        for corner, source_corner in enumerate(source_corner_for_target):
            source_uv_triangles[target_face_index, corner] = source_uv[
                source_face[source_corner]
            ]
    indexed_geometry_is_identity = (
        source.vertices.shape == target.vertices.shape
        and np.array_equal(source.faces, target.faces)
    )
    uv_is_identity = (
        indexed_geometry_is_identity
        and source_uv.shape == target_uv.shape
        and np.allclose(source_uv, target_uv, atol=1e-6)
    )
    if correspondence == "identity" and not uv_is_identity:
        raise ValueError("Expected identical source and target UV indexing")
    if resolution <= 0:
        raise ValueError("resolution must be positive")

    source_textures = extract_pbr_textures(source_asset)
    if uv_is_identity:
        textures = {
            channel: image.resize(
                (resolution, resolution),
                Image.Resampling.BILINEAR,
            )
            for channel, image in source_textures.items()
        }
        prepared_mask = Path(target_mesh).parent / "uv_mask.png"
        if prepared_mask.is_file():
            uv_mask = Image.open(prepared_mask).convert("L").resize(
                (resolution, resolution), Image.Resampling.NEAREST
            )
            mask_array = np.asarray(uv_mask) > 0
            return UVTextureTransfer(
                textures=textures,
                uv_mask=uv_mask,
                correspondence="identity",
                valid_fraction=float(mask_array.mean()),
            )

    outputs = {
        channel: np.zeros(
            (resolution, resolution, 3 if channel == "albedo" else 1),
            dtype=np.float32,
        )
        for channel in source_textures
    }
    mask = np.zeros((resolution, resolution), dtype=bool)

    for face_index, target_face in enumerate(target.faces):
        target_triangle = np.asarray(target_uv[target_face], dtype=np.float64)
        pixels = target_triangle.copy()
        pixels[:, 0] = pixels[:, 0] * resolution - 0.5
        pixels[:, 1] = (1.0 - pixels[:, 1]) * resolution - 0.5
        minimum = np.maximum(np.floor(pixels.min(axis=0)).astype(int), 0)
        maximum = np.minimum(
            np.ceil(pixels.max(axis=0)).astype(int), resolution - 1
        )
        if np.any(maximum < minimum):
            continue
        xs, ys = np.meshgrid(
            np.arange(minimum[0], maximum[0] + 1),
            np.arange(minimum[1], maximum[1] + 1),
        )
        points = np.stack([xs.ravel(), ys.ravel()], axis=1)
        a, b, c = pixels
        denominator = (b[1] - c[1]) * (a[0] - c[0]) + (
            c[0] - b[0]
        ) * (a[1] - c[1])
        if abs(denominator) < 1e-12:
            continue
        wa = (
            (b[1] - c[1]) * (points[:, 0] - c[0])
            + (c[0] - b[0]) * (points[:, 1] - c[1])
        ) / denominator
        wb = (
            (c[1] - a[1]) * (points[:, 0] - c[0])
            + (a[0] - c[0]) * (points[:, 1] - c[1])
        ) / denominator
        wc = 1.0 - wa - wb
        inside = (wa >= -1e-7) & (wb >= -1e-7) & (wc >= -1e-7)
        if not inside.any():
            continue
        points = points[inside]
        barycentric = np.stack([wa[inside], wb[inside], wc[inside]], axis=1)
        sampled_uv = barycentric @ source_uv_triangles[face_index]
        rows = points[:, 1]
        columns = points[:, 0]
        for channel, image in source_textures.items():
            outputs[channel][rows, columns] = _sample_bilinear(image, sampled_uv)
        mask[rows, columns] = True

    images: dict[PBRChannel, Image.Image] = {}
    for channel, array in outputs.items():
        encoded = np.round(np.clip(array, 0.0, 1.0) * 255.0).astype(np.uint8)
        if channel == "albedo":
            images[channel] = Image.fromarray(encoded, "RGB")
        else:
            images[channel] = Image.fromarray(encoded[..., 0], "L")
    return UVTextureTransfer(
        textures=images,
        uv_mask=Image.fromarray(mask.astype(np.uint8) * 255, "L"),
        correspondence="identity" if uv_is_identity else "topology",
        valid_fraction=float(mask.mean()),
    )


def create_textured_glb(
    target_mesh: str | Path,
    textures: Mapping[PBRChannel, Image.Image | str | Path],
    output_path: str | Path,
) -> Path:
    """Create a self-contained glTF metallic-roughness GLB on target UVs."""
    import trimesh

    missing = {"albedo", "roughness", "metallic"} - set(textures)
    if missing:
        raise ValueError(f"Missing PBR textures: {sorted(missing)}")

    def image(name: PBRChannel, mode: str) -> Image.Image:
        value = textures[name]
        source = value if isinstance(value, Image.Image) else Image.open(value)
        return source.convert(mode)

    mesh = _load_mesh(target_mesh)
    uv = getattr(mesh.visual, "uv", None)
    if uv is None:
        raise ValueError(f"Target mesh has no UV coordinates: {target_mesh}")
    albedo = image("albedo", "RGB")
    roughness = image("roughness", "L")
    metallic = image("metallic", "L")
    if roughness.size != albedo.size or metallic.size != albedo.size:
        raise ValueError("Canonical PBR textures must have matching dimensions")
    zeros = Image.new("L", albedo.size, 0)
    metallic_roughness = Image.merge("RGB", (zeros, roughness, metallic))
    material = trimesh.visual.material.PBRMaterial(
        baseColorTexture=albedo,
        metallicRoughnessTexture=metallic_roughness,
        roughnessFactor=1.0,
        metallicFactor=1.0,
    )
    mesh.visual = trimesh.visual.TextureVisuals(
        uv=np.asarray(uv).copy(), material=material
    )
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    mesh.export(output_path, file_type="glb")
    return output_path
