"""Blender-side mesh export and light-conditioned appearance baking."""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
from pathlib import Path

import bpy
from mathutils import Matrix, Vector


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


def configure_cycles(config: dict) -> None:
    scene = bpy.context.scene
    scene.render.engine = "CYCLES"
    scene.cycles.samples = int(config["samples_per_pixel"])
    scene.cycles.use_denoising = bool(config.get("denoise", True))
    scene.cycles.seed = 0
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGBA"
    scene.render.image_settings.color_depth = "8"
    scene.view_settings.view_transform = "Standard"
    scene.view_settings.look = "None"
    scene.view_settings.exposure = 0.0
    scene.view_settings.gamma = 1.0
    scene.render.bake.margin = int(config.get("bake_margin", 16))

    if str(config.get("device", "cpu")).lower() == "cpu":
        scene.cycles.device = "CPU"
        return
    try:
        preferences = bpy.context.preferences.addons["cycles"].preferences
        preferences.compute_device_type = str(config["device"]).upper()
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


def setup_environment(light: dict, view_yaw_deg: float) -> float:
    world = bpy.context.scene.world or bpy.data.worlds.new("BenchmarkWorld")
    bpy.context.scene.world = world
    world.use_nodes = True
    nodes = world.node_tree.nodes
    links = world.node_tree.links
    nodes.clear()
    texture = nodes.new(type="ShaderNodeTexEnvironment")
    texture.image = bpy.data.images.load(light["path"], check_existing=True)
    mapping = nodes.new(type="ShaderNodeMapping")
    rotation_deg = float(light["rotation_deg"]) + view_yaw_deg
    mapping.inputs["Rotation"].default_value[2] = math.radians(rotation_deg)
    coordinates = nodes.new(type="ShaderNodeTexCoord")
    background = nodes.new(type="ShaderNodeBackground")
    background.inputs["Strength"].default_value = float(light["strength"])
    output = nodes.new(type="ShaderNodeOutputWorld")
    links.new(coordinates.outputs["Generated"], mapping.inputs["Vector"])
    links.new(mapping.outputs["Vector"], texture.inputs["Vector"])
    links.new(texture.outputs["Color"], background.inputs["Color"])
    links.new(background.outputs["Background"], output.inputs["Surface"])
    return rotation_deg


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


def select_meshes(meshes: list[bpy.types.Object]) -> None:
    bpy.ops.object.select_all(action="DESELECT")
    for mesh in meshes:
        mesh.select_set(True)
    bpy.context.view_layer.objects.active = meshes[0]


def bake_texture(path: Path, meshes: list[bpy.types.Object], resolution: int) -> None:
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


def main() -> None:
    job = json.loads(arguments().job.read_text())
    output_dir = Path(job["output_dir"])
    clear_scene()
    imported, meshes = import_asset(Path(job["asset_path"]))
    normalization = normalize_asset(imported, meshes)
    configure_cycles(job["renderer"])
    resolution = int(
        job["renderer"].get("texture_resolution", job["renderer"]["resolution"])
    )

    for view in job["views"]:
        view_dir = output_dir / view["id"]
        mesh_path = view_dir / "mesh.obj"
        metadata_path = view_dir / "metadata.json"
        texture_paths = [
            view_dir / "textures" / f"{light['id']}.png" for light in job["lights"]
        ]
        expected = [mesh_path, metadata_path, *texture_paths]
        if all(path.is_file() for path in expected) and not job["overwrite"]:
            print(f"skip complete {job['object_id']}/{view['id']}")
            continue

        export_obj(mesh_path, meshes)
        baked_lights = []
        for light, texture_path in zip(job["lights"], texture_paths):
            rotation_deg = setup_environment(light, float(view["yaw_deg"]))
            bake_texture(texture_path, meshes, resolution)
            baked_lights.append(
                {
                    "id": light["id"],
                    "rotation_deg": rotation_deg,
                    "strength": light["strength"],
                    "texture": f"textures/{light['id']}.png",
                }
            )

        metadata = {
            "schema_version": 2,
            "representation": "3d",
            "object_id": job["object_id"],
            "asset_path": job["asset_path"],
            "view_id": view["id"],
            "yaw_deg": view["yaw_deg"],
            "mesh": "mesh.obj",
            "texture_resolution": [resolution, resolution],
            "normalization_source_to_world": matrix_rows(normalization),
            "lights": baked_lights,
            "renderer": job["renderer"],
            "blender_version": bpy.app.version_string,
        }
        metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")


if __name__ == "__main__":
    main()
