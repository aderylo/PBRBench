"""Blender-side mesh export and light-conditioned appearance baking."""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
import traceback
from pathlib import Path

import bpy
import numpy as np
from mathutils import Matrix, Vector

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.preprocessing.utils import (  # noqa: E402
    BakeJob,
    LightSpec,
    ObjectMetadata,
    RendererSpec,
    completion_marker,
    expected_paths,
    recipe_hash,
)

NORMAL_CONVENTION = {
    "space": "tangent",
    "encoding": "rgb = normal * 0.5 + 0.5; neutral (flat) texel = (0.5, 0.5, 1.0)",
    "includes_material_normal_maps": True,
    "note": (
        "Interpret against a MikkTSpace-compatible tangent basis computed from "
        "mesh.obj with the same UVs."
    ),
}

ALPHA_CONVENTION = {
    "albedo.png": "material opacity (Principled Alpha) inside UV coverage; 0 outside",
    "roughness.png": "UV validity mask (1 = valid baked texel, 0 = no coverage)",
    "metallic.png": "UV validity mask (1 = valid baked texel, 0 = no coverage)",
    "normal.png": "UV validity mask (1 = valid baked texel, 0 = no coverage)",
    "uv_mask.png": (
        "binary UV validity mask (white = valid); includes bake margin dilation"
    ),
    "textures/<light_id>.png": "lit appearance; alpha is the baked material alpha",
}


def capture_color_encoding() -> dict:
    """Document the image settings actually used for the saved PNG files."""
    scene = bpy.context.scene
    settings = scene.render.image_settings
    return {
        "file_format": settings.file_format,
        "color_mode": settings.color_mode,
        "bit_depth": int(settings.color_depth),
        "view_transform": scene.view_settings.view_transform,
        "gamma": float(scene.view_settings.gamma),
        "exposure": float(scene.view_settings.exposure),
        "pixel_values": "scene-linear (no sRGB transfer function applied)",
    }


def arguments() -> argparse.Namespace:
    argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    parser = argparse.ArgumentParser()
    parser.add_argument("--job", type=Path, required=True)
    return parser.parse_args(argv)


def matrix_rows(matrix: Matrix) -> list[list[float]]:
    return [[float(value) for value in row] for row in matrix]


def clear_scene() -> None:
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)


def import_asset(path: Path) -> tuple[list[bpy.types.Object], list[bpy.types.Object]]:
    before = set(bpy.context.scene.objects)
    bpy.ops.import_scene.gltf(filepath=str(path))
    imported = [item for item in bpy.context.scene.objects if item not in before]
    for item in [item for item in imported if item.type in {"CAMERA", "LIGHT"}]:
        bpy.data.objects.remove(item, do_unlink=True)
        imported.remove(item)
    meshes = [item for item in imported if item.type == "MESH"]
    if not meshes:
        raise RuntimeError(f"No mesh objects imported from {path}")
    if any(not mesh.data.uv_layers for mesh in meshes):
        missing = ", ".join(mesh.name for mesh in meshes if not mesh.data.uv_layers)
        raise RuntimeError(f"Cannot bake meshes without UV coordinates: {missing}")
    return imported, meshes


def bounds(meshes: list[bpy.types.Object]) -> tuple[Vector, float]:
    bpy.context.view_layer.update()
    points = [
        mesh.matrix_world @ Vector(corner)
        for mesh in meshes
        for corner in mesh.bound_box
    ]
    minimum = Vector(tuple(min(point[axis] for point in points) for axis in range(3)))
    maximum = Vector(tuple(max(point[axis] for point in points) for axis in range(3)))
    center = (minimum + maximum) * 0.5
    radius = max((point - center).length for point in points)
    if radius <= 0:
        raise RuntimeError("Imported asset has an empty bounding sphere")
    return center, radius


def normalize_asset(
    imported: list[bpy.types.Object], meshes: list[bpy.types.Object]
) -> Matrix:
    center, radius = bounds(meshes)
    transform = Matrix.Scale(0.5 / radius, 4) @ Matrix.Translation(-center)
    imported_set = set(imported)
    for item in [item for item in imported if item.parent not in imported_set]:
        item.matrix_world = transform @ item.matrix_world
    bpy.context.view_layer.update()
    return transform


def configure_cycles(config: RendererSpec) -> None:
    scene = bpy.context.scene
    scene.render.engine = "CYCLES"
    scene.cycles.samples = config.samples_per_pixel
    scene.cycles.use_denoising = config.denoise
    scene.cycles.seed = 0
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGBA"
    scene.render.image_settings.color_depth = "8"
    scene.view_settings.view_transform = "Standard"
    scene.view_settings.look = "None"
    scene.view_settings.exposure = 0.0
    scene.view_settings.gamma = 1.0
    scene.render.bake.margin = config.bake_margin

    if config.device.lower() == "cpu":
        scene.cycles.device = "CPU"
        return
    try:
        preferences = bpy.context.preferences.addons["cycles"].preferences
        preferences.compute_device_type = config.device.upper()
        preferences.get_devices()
        usable = [device for device in preferences.devices if device.type != "CPU"]
        if not usable:
            raise RuntimeError("no GPU device reported by Cycles")
        for device in preferences.devices:
            device.use = device.type != "CPU"
        scene.cycles.device = "GPU"
    except Exception as error:
        logging.warning("Cycles GPU unavailable (%s); using CPU", error)
        scene.cycles.device = "CPU"


def setup_environment(light: LightSpec) -> None:
    world = bpy.context.scene.world or bpy.data.worlds.new("BenchmarkWorld")
    bpy.context.scene.world = world
    world.use_nodes = True
    nodes = world.node_tree.nodes
    links = world.node_tree.links
    nodes.clear()
    texture = nodes.new(type="ShaderNodeTexEnvironment")
    texture.image = bpy.data.images.load(light.path, check_existing=True)
    mapping = nodes.new(type="ShaderNodeMapping")
    mapping.inputs["Rotation"].default_value[2] = math.radians(light.rotation_deg)
    coordinates = nodes.new(type="ShaderNodeTexCoord")
    background = nodes.new(type="ShaderNodeBackground")
    background.inputs["Strength"].default_value = light.strength
    output = nodes.new(type="ShaderNodeOutputWorld")
    links.new(coordinates.outputs["Generated"], mapping.inputs["Vector"])
    links.new(mapping.outputs["Vector"], texture.inputs["Vector"])
    links.new(texture.outputs["Color"], background.inputs["Color"])
    links.new(background.outputs["Background"], output.inputs["Surface"])


def materials(meshes: list[bpy.types.Object]) -> list[bpy.types.Material]:
    found = []
    seen = set()
    for mesh in meshes:
        for material in mesh.data.materials:
            if material and material.name_full not in seen:
                material.use_nodes = True
                found.append(material)
                seen.add(material.name_full)
    if not found:
        raise RuntimeError("Imported asset has no materials to bake")
    return found


def validate_materials(found: list[bpy.types.Material]) -> None:
    """Fail loudly on materials that cannot produce trustworthy PBR ground truth."""
    unsupported = [
        material.name_full
        for material in found
        if not any(node.type == "BSDF_PRINCIPLED" for node in material.node_tree.nodes)
    ]
    if unsupported:
        raise RuntimeError(
            "Materials without a Principled BSDF cannot produce trustworthy PBR ground "
            "truth; refusing to bake: " + ", ".join(unsupported)
        )
    for material in found:
        principled = next(
            (
                node
                for node in material.node_tree.nodes
                if node.type == "BSDF_PRINCIPLED"
            ),
            None,
        )
        transmission = principled.inputs.get("Transmission Weight")
        if transmission is None:
            transmission = principled.inputs.get("Transmission")
        if transmission is not None and float(transmission.default_value) > 0.0:
            logging.warning(
                "Material '%s' uses transmission (glass-like); its albedo GT is the "
                "dark base color and may be misleading",
                material.name_full,
            )


def select_meshes(meshes: list[bpy.types.Object]) -> None:
    bpy.ops.object.select_all(action="DESELECT")
    for mesh in meshes:
        mesh.select_set(True)
    bpy.context.view_layer.objects.active = meshes[0]


def bake_texture(
    path: Path, meshes: list[bpy.types.Object], resolution: int
) -> None:
    image = bpy.data.images.new(
        f"BenchmarkBake_{path.stem}", width=resolution, height=resolution, alpha=True
    )
    image.generated_color = (0.0, 0.0, 0.0, 0.0)
    bake_nodes = []
    for material in materials(meshes):
        node = material.node_tree.nodes.new(type="ShaderNodeTexImage")
        node.name = "BenchmarkBakeTarget"
        node.image = image
        material.node_tree.nodes.active = node
        node.select = True
        bake_nodes.append((material, node))

    select_meshes(meshes)
    bpy.ops.object.bake(type="COMBINED", use_clear=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    image.filepath_raw = str(path)
    image.file_format = "PNG"
    image.save_render(str(path), scene=bpy.context.scene)

    for material, node in bake_nodes:
        material.node_tree.nodes.remove(node)
    bpy.data.images.remove(image)


def channel_source(material: bpy.types.Material, channel: str):
    nodes = material.node_tree.nodes
    principled = next((node for node in nodes if node.type == "BSDF_PRINCIPLED"), None)
    if principled is None:
        return None, None
    socket_name = {
        "albedo": "Base Color",
        "roughness": "Roughness",
        "metallic": "Metallic",
        "alpha": "Alpha",
    }[channel]
    socket = principled.inputs.get(socket_name)
    if socket is None:
        return None, None
    if socket.is_linked:
        return socket.links[0].from_socket, None
    return None, socket.default_value


def set_material_channel(material: bpy.types.Material, channel: str) -> None:
    tree = material.node_tree
    nodes = tree.nodes
    links = tree.links
    output = next(
        (
            node
            for node in nodes
            if node.type == "OUTPUT_MATERIAL" and node.is_active_output
        ),
        None,
    )
    if output is None:
        output = nodes.new(type="ShaderNodeOutputMaterial")
    for link in list(output.inputs["Surface"].links):
        links.remove(link)
    emission = nodes.get("BenchmarkChannelEmission") or nodes.new(
        type="ShaderNodeEmission"
    )
    emission.name = "BenchmarkChannelEmission"
    source, default = channel_source(material, channel)
    for link in list(emission.inputs["Color"].links):
        links.remove(link)
    if source is not None:
        links.new(source, emission.inputs["Color"])
    elif default is not None:
        if hasattr(default, "__len__"):
            emission.inputs["Color"].default_value = tuple(default)
        else:
            value = float(default)
            emission.inputs["Color"].default_value = (value, value, value, 1.0)
    else:
        fallback = {
            "albedo": material.diffuse_color,
            "alpha": 1.0,
            "roughness": 1.0,
            "metallic": 0.0,
        }[channel]
        if hasattr(fallback, "__len__"):
            emission.inputs["Color"].default_value = tuple(fallback)
        else:
            emission.inputs["Color"].default_value = (fallback, fallback, fallback, 1.0)
    links.new(emission.outputs["Emission"], output.inputs["Surface"])


def save_material_output(material: bpy.types.Material):
    tree = material.node_tree
    output = next(
        (node for node in tree.nodes if node.type == "OUTPUT_MATERIAL" and node.is_active_output),
        None,
    )
    if output and output.inputs["Surface"].links:
        return output.inputs["Surface"].links[0].from_socket
    return None


def restore_material_output(material: bpy.types.Material, original_socket) -> None:
    tree = material.node_tree
    output = next(
        (node for node in tree.nodes if node.type == "OUTPUT_MATERIAL" and node.is_active_output),
        None,
    )
    if output is None:
        return
    for link in list(output.inputs["Surface"].links):
        tree.links.remove(link)
    if original_socket is not None:
        tree.links.new(original_socket, output.inputs["Surface"])
    emission = tree.nodes.get("BenchmarkChannelEmission")
    if emission is not None:
        tree.nodes.remove(emission)


def bake_channel(
    path: Path,
    meshes: list[bpy.types.Object],
    channel: str,
    resolution: int,
    mask_image: bpy.types.Image | None = None,
    alpha_source: bpy.types.Image | None = None,
) -> None:
    saved_sockets = {}
    for material in materials(meshes):
        saved_sockets[material] = save_material_output(material)
        set_material_channel(material, channel)

    image = bpy.data.images.new(
        f"BenchmarkBake_{channel}_{path.stem}", width=resolution, height=resolution, alpha=True
    )
    image.generated_color = (0.0, 0.0, 0.0, 0.0)
    bake_nodes = []
    for material in materials(meshes):
        node = material.node_tree.nodes.new(type="ShaderNodeTexImage")
        node.name = "BenchmarkBakeTarget"
        node.image = image
        material.node_tree.nodes.active = node
        node.select = True
        bake_nodes.append((material, node))

    select_meshes(meshes)
    bpy.ops.object.bake(type="EMIT", use_clear=True)

    if mask_image is not None:
        covered = coverage(mask_image)
        width, height = image.size
        pixels = np.empty(width * height * 4, dtype=np.float32)
        image.pixels.foreach_get(pixels)
        if alpha_source is not None:
            alpha_width, alpha_height = alpha_source.size
            alpha_pixels = np.empty(alpha_width * alpha_height * 4, dtype=np.float32)
            alpha_source.pixels.foreach_get(alpha_pixels)
            pixels[3::4] = np.where(covered, alpha_pixels[0::4], 0.0)
        else:
            pixels[3::4] = np.where(covered, 1.0, 0.0)
        pixels[0::4] = np.where(covered, pixels[0::4], 0.0)
        pixels[1::4] = np.where(covered, pixels[1::4], 0.0)
        pixels[2::4] = np.where(covered, pixels[2::4], 0.0)
        image.pixels.foreach_set(pixels)

    path.parent.mkdir(parents=True, exist_ok=True)
    image.filepath_raw = str(path)
    image.file_format = "PNG"
    image.save_render(str(path), scene=bpy.context.scene)

    for material, node in bake_nodes:
        material.node_tree.nodes.remove(node)
    bpy.data.images.remove(image)

    for material, saved_socket in saved_sockets.items():
        restore_material_output(material, saved_socket)


def bake_normal(
    path: Path,
    meshes: list[bpy.types.Object],
    resolution: int,
    mask_image: bpy.types.Image | None = None,
) -> None:
    image = bpy.data.images.new(
        f"BenchmarkBake_normal_{path.stem}", width=resolution, height=resolution, alpha=True
    )
    image.generated_color = (0.5, 0.5, 1.0, 1.0)
    bake_nodes = []
    for material in materials(meshes):
        node = material.node_tree.nodes.new(type="ShaderNodeTexImage")
        node.name = "BenchmarkBakeTarget"
        node.image = image
        material.node_tree.nodes.active = node
        node.select = True
        bake_nodes.append((material, node))

    select_meshes(meshes)
    bpy.ops.object.bake(type="NORMAL", normal_space="TANGENT", use_clear=True)

    if mask_image is not None:
        covered = coverage(mask_image)
        width, height = image.size
        pixels = np.empty(width * height * 4, dtype=np.float32)
        image.pixels.foreach_get(pixels)
        pixels[0::4] = np.where(covered, pixels[0::4], 0.5)
        pixels[1::4] = np.where(covered, pixels[1::4], 0.5)
        pixels[2::4] = np.where(covered, pixels[2::4], 1.0)
        pixels[3::4] = np.where(covered, 1.0, 0.0)
        image.pixels.foreach_set(pixels)

    path.parent.mkdir(parents=True, exist_ok=True)
    image.filepath_raw = str(path)
    image.file_format = "PNG"
    image.save_render(str(path), scene=bpy.context.scene)

    for material, node in bake_nodes:
        material.node_tree.nodes.remove(node)
    bpy.data.images.remove(image)


def export_obj(path: Path, meshes: list[bpy.types.Object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    select_meshes(meshes)
    if hasattr(bpy.ops.wm, "obj_export"):
        bpy.ops.wm.obj_export(
            filepath=str(path),
            export_selected_objects=True,
            export_materials=False,
            forward_axis="NEGATIVE_Z",
            up_axis="Y",
        )
    else:
        bpy.ops.export_scene.obj(
            filepath=str(path),
            use_selection=True,
            use_materials=False,
            axis_forward="-Z",
            axis_up="Y",
        )


def coverage(mask_image: bpy.types.Image) -> np.ndarray:
    """Boolean array of texels that received baked content."""
    width, height = mask_image.size
    pixels = np.empty(width * height * 4, dtype=np.float32)
    mask_image.pixels.foreach_get(pixels)
    return (
        (pixels[0::4] > 0.0)
        | (pixels[1::4] > 0.0)
        | (pixels[2::4] > 0.0)
        | (pixels[3::4] > 0.0)
    )


def bake_uv_mask(
    meshes: list[bpy.types.Object], resolution: int
) -> bpy.types.Image:
    """Bake a material-independent UV pass whose alpha marks covered texels.

    The returned image is kept in memory so channel bakes can consume it
    without a load/save round trip.
    """
    image = bpy.data.images.new(
        "BenchmarkUVMask", width=resolution, height=resolution, alpha=True
    )
    image.generated_color = (0.0, 0.0, 0.0, 0.0)
    bake_nodes = []
    for material in materials(meshes):
        node = material.node_tree.nodes.new(type="ShaderNodeTexImage")
        node.name = "BenchmarkBakeTarget"
        node.image = image
        material.node_tree.nodes.active = node
        node.select = True
        bake_nodes.append((material, node))

    select_meshes(meshes)
    bpy.ops.object.bake(type="UV", use_clear=True)

    for material, node in bake_nodes:
        material.node_tree.nodes.remove(node)
    return image


def save_binary_mask(path: Path, mask_image: bpy.types.Image) -> None:
    """Save a white-on-black PNG of the texels that received baked content.

    The mask includes the configured bake margin (dilated coverage), matching
    the extent of the baked channel textures.
    """
    width, height = mask_image.size
    covered = coverage(mask_image)
    pixels = np.zeros(width * height * 4, dtype=np.float32)
    for index in range(3):
        pixels[index::4] = np.where(covered, 1.0, 0.0)
    pixels[3::4] = 1.0
    image = bpy.data.images.new(
        "BenchmarkBinaryMask", width=width, height=height, alpha=True
    )
    image.pixels.foreach_set(pixels)
    path.parent.mkdir(parents=True, exist_ok=True)
    image.filepath_raw = str(path)
    image.file_format = "PNG"
    image.save_render(str(path), scene=bpy.context.scene)
    bpy.data.images.remove(image)


def bake_material_alpha(
    meshes: list[bpy.types.Object], resolution: int
) -> bpy.types.Image:
    """Bake the Principled Alpha (material opacity) into an in-memory image."""
    saved_sockets = {}
    for material in materials(meshes):
        saved_sockets[material] = save_material_output(material)
        set_material_channel(material, "alpha")

    image = bpy.data.images.new(
        "BenchmarkAlphaBake", width=resolution, height=resolution, alpha=True
    )
    image.generated_color = (0.0, 0.0, 0.0, 0.0)
    bake_nodes = []
    for material in materials(meshes):
        node = material.node_tree.nodes.new(type="ShaderNodeTexImage")
        node.name = "BenchmarkBakeTarget"
        node.image = image
        material.node_tree.nodes.active = node
        node.select = True
        bake_nodes.append((material, node))

    select_meshes(meshes)
    bpy.ops.object.bake(type="EMIT", use_clear=True)

    for material, node in bake_nodes:
        material.node_tree.nodes.remove(node)

    for material, saved_socket in saved_sockets.items():
        restore_material_output(material, saved_socket)
    return image


def check_source_texture_resolution(
    job: BakeJob, target_resolution: int
) -> None:
    """Warn if any source asset texture is of lower resolution than target bake resolution."""
    native_sizes = [
        (img.name, img.size[0], img.size[1])
        for img in bpy.data.images
        if img.size[0] > 0 and not img.name.startswith("Benchmark")
    ]
    for name, w, h in native_sizes:
        if w < target_resolution or h < target_resolution:
            msg = (
                f"WARNING: [{job.object_id}] Source texture '{name}' resolution ({w}x{h}) "
                f"is lower than target bake resolution ({target_resolution}x{target_resolution}). Upsampling will occur."
            )
            print(msg)
            logging.warning(msg)


def metadata_matches_recipe(metadata_path: Path, job: BakeJob) -> bool:
    """Check whether stored metadata was produced with the current settings."""
    try:
        payload = json.loads(metadata_path.read_text())
    except (OSError, ValueError):
        return False
    return payload.get("recipe_hash") == recipe_hash(job)


def write_atomic(path: Path, content: str) -> None:
    """Write a file via a temporary sibling so a crash never leaves partial data."""
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(content)
    temporary.replace(path)


def prepare(job: BakeJob, output_dir: Path) -> None:
    clear_scene()
    imported, meshes = import_asset(Path(job.asset_path))
    normalization = normalize_asset(imported, meshes)
    configure_cycles(job.renderer)
    target_res = job.renderer.texture_resolution
    if target_res is None:
        target_res = 1024
        logging.warning(
            "texture_resolution is not configured; falling back to %d", target_res
        )
    check_source_texture_resolution(job, target_res)
    validate_materials(materials(meshes))

    mesh_path = output_dir / "mesh.obj"
    metadata_path = output_dir / "metadata.json"
    marker_path = completion_marker(output_dir)
    texture_paths = [output_dir / "textures" / f"{light.id}.png" for light in job.lights]
    pbr_paths = [
        output_dir / "pbr" / f"{channel}.png"
        for channel in ("albedo", "roughness", "metallic", "normal")
    ]

    if (
        not job.overwrite
        and marker_path.is_file()
        and metadata_matches_recipe(metadata_path, job)
        and all(path.is_file() for path in expected_paths(job))
    ):
        print(f"skip complete {job.object_id}")
        return

    # Remove completion signals first; a failed rebake must not look valid.
    marker_path.unlink(missing_ok=True)
    metadata_path.unlink(missing_ok=True)
    (output_dir / "error.txt").unlink(missing_ok=True)
    export_obj(mesh_path, meshes)

    mask_image = bake_uv_mask(meshes, target_res)
    save_binary_mask(output_dir / "uv_mask.png", mask_image)

    alpha_image = bake_material_alpha(meshes, target_res)
    bake_channel(
        pbr_paths[0],
        meshes,
        "albedo",
        target_res,
        mask_image=mask_image,
        alpha_source=alpha_image,
    )
    bpy.data.images.remove(alpha_image)

    for channel, pbr_path in zip(("roughness", "metallic"), pbr_paths[1:3]):
        bake_channel(pbr_path, meshes, channel, target_res, mask_image=mask_image)
    bake_normal(pbr_paths[3], meshes, target_res, mask_image=mask_image)
    bpy.data.images.remove(mask_image)

    # Bake lit appearance under environment lights
    for light, texture_path in zip(job.lights, texture_paths):
        setup_environment(light)
        bake_texture(texture_path, meshes, target_res)

    metadata = ObjectMetadata(
        asset_path=job.asset_path,
        normalization_source_to_world=matrix_rows(normalization),
        recipe_hash=recipe_hash(job),
        blender_version=bpy.app.version_string,
        materials=[material.name_full for material in materials(meshes)],
        color_encoding=capture_color_encoding(),
        normal_convention=NORMAL_CONVENTION,
        alpha_convention=ALPHA_CONVENTION,
    )
    write_atomic(metadata_path, json.dumps(metadata.to_dict(), indent=2) + "\n")
    marker_path.touch()


def main() -> None:
    job = BakeJob.from_dict(json.loads(arguments().job.read_text()))
    output_dir = Path(job.output_dir)
    try:
        prepare(job, output_dir)
    except Exception as error:
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "error.txt").write_text(
            f"{type(error).__name__}: {error}\n{traceback.format_exc()}"
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
