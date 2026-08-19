"""Blender-side relighting script for 3D GLB/OBJ mesh evaluation."""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import bpy  # noqa: E402
from mathutils import Matrix, Vector  # noqa: E402

from src.data.preprocessing.utils import (  # noqa: E402
    LightSpec,
    RendererSpec,
)


def arguments() -> argparse.Namespace:
    argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    parser = argparse.ArgumentParser()
    parser.add_argument("--job", type=Path, required=True)
    return parser.parse_args(argv)


def clear_scene() -> None:
    for obj in list(bpy.data.objects):
        bpy.data.objects.remove(obj, do_unlink=True)
    for mesh in list(bpy.data.meshes):
        bpy.data.meshes.remove(mesh, do_unlink=True)
    for mat in list(bpy.data.materials):
        bpy.data.materials.remove(mat, do_unlink=True)
    for img in list(bpy.data.images):
        bpy.data.images.remove(img, do_unlink=True)
    for cam in list(bpy.data.cameras):
        bpy.data.cameras.remove(cam, do_unlink=True)
    for light in list(bpy.data.lights):
        bpy.data.lights.remove(light, do_unlink=True)
    for tex in list(bpy.data.textures):
        bpy.data.textures.remove(tex, do_unlink=True)
    for world in list(bpy.data.worlds):
        bpy.data.worlds.remove(world, do_unlink=True)
    try:
        bpy.data.orphans_purge(do_local_ids=True, do_linked_ids=True, do_recursive=True)
    except Exception:
        pass


def import_asset(path: Path) -> tuple[list[bpy.types.Object], list[bpy.types.Object]]:
    before = set(bpy.context.scene.objects)
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix in (".glb", ".gltf"):
        bpy.ops.import_scene.gltf(filepath=str(path))
    elif suffix == ".obj":
        if hasattr(bpy.ops.wm, "obj_import"):
            bpy.ops.wm.obj_import(filepath=str(path))
        else:
            bpy.ops.import_scene.obj(filepath=str(path))
    else:
        bpy.ops.import_scene.gltf(filepath=str(path))

    imported = [item for item in bpy.context.scene.objects if item not in before]
    for item in [item for item in imported if item.type in {"CAMERA", "LIGHT"}]:
        bpy.data.objects.remove(item, do_unlink=True)
        if item in imported:
            imported.remove(item)
    meshes = [item for item in imported if item.type == "MESH"]
    if not meshes:
        raise RuntimeError(f"No mesh objects imported from {path}")
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
        radius = 0.5
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


def apply_normalization(imported: list[bpy.types.Object], rows: list[list[float]]) -> None:
    transform = Matrix(rows)
    imported_set = set(imported)
    for item in [item for item in imported if item.parent not in imported_set]:
        item.matrix_world = transform @ item.matrix_world
    bpy.context.view_layer.update()


def configure_render(config: RendererSpec) -> None:
    scene = bpy.context.scene
    scene.render.engine = "CYCLES"
    scene.cycles.samples = config.samples_per_pixel
    scene.cycles.use_denoising = config.denoise
    scene.cycles.seed = 0
    scene.render.resolution_x = config.resolution
    scene.render.resolution_y = config.resolution
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGBA"
    scene.render.image_settings.color_depth = "8"
    scene.render.film_transparent = config.transparent_background
    scene.view_settings.view_transform = "Standard"
    scene.view_settings.look = "None"
    scene.view_settings.exposure = 0.0
    scene.view_settings.gamma = 1.0

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


def add_camera(camera_data: dict | None = None) -> bpy.types.Object:
    bpy.ops.object.camera_add()
    camera = bpy.context.object
    camera.name = "Benchmark3DCamera"
    camera.data.sensor_width = 36.0
    camera.data.sensor_height = 36.0

    if camera_data and "camera_to_world" in camera_data and "intrinsics" in camera_data:
        width = int(camera_data["resolution"][0]) if "resolution" in camera_data else 1024
        camera.data.lens = float(camera_data["intrinsics"][0][0]) * 36.0 / width
        camera.matrix_world = Matrix(camera_data["camera_to_world"])
    else:
        camera.data.lens = 50.0
        location = Vector((0.0, -2.2, 0.5))
        target = Vector((0.0, 0.0, 0.0))
        direction = target - location
        rot_quat = direction.to_track_quat("-Z", "Y")
        camera.location = location
        camera.rotation_euler = rot_quat.to_euler()

    bpy.context.scene.camera = camera
    bpy.context.view_layer.update()
    return camera


def set_camera(camera: bpy.types.Object, camera_data: dict | None = None) -> None:
    if camera_data and "camera_to_world" in camera_data and "intrinsics" in camera_data:
        width = int(camera_data["resolution"][0]) if "resolution" in camera_data else 1024
        camera.data.lens = float(camera_data["intrinsics"][0][0]) * 36.0 / width
        camera.matrix_world = Matrix(camera_data["camera_to_world"])
    else:
        camera.data.lens = 50.0
        location = Vector((0.0, -2.2, 0.5))
        target = Vector((0.0, 0.0, 0.0))
        direction = target - location
        rot_quat = direction.to_track_quat("-Z", "Y")
        camera.location = location
        camera.rotation_euler = rot_quat.to_euler()
    bpy.context.view_layer.update()


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


def render_png(output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    bpy.context.scene.render.filepath = str(output_path)
    bpy.ops.render.render(write_still=True)


def bake_combined_texture(mesh: bpy.types.Object, output_path: Path, resolution: int = 1024) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    bpy.ops.object.select_all(action="DESELECT")
    mesh.select_set(True)
    bpy.context.view_layer.objects.active = mesh

    image = bpy.data.images.new("BakeTarget", width=resolution, height=resolution)
    for mat_slot in mesh.material_slots:
        if mat_slot.material and mat_slot.material.node_tree:
            node = mat_slot.material.node_tree.nodes.new("ShaderNodeTexImage")
            node.image = image
            mat_slot.material.node_tree.nodes.active = node

    scene = bpy.context.scene
    scene.render.bake.use_pass_direct = True
    scene.render.bake.use_pass_indirect = True
    scene.render.bake.use_pass_color = True
    bpy.ops.object.bake(type="COMBINED")
    image.filepath_raw = str(output_path)
    image.file_format = "PNG"
    image.save()


def main() -> None:
    job = json.loads(arguments().job.read_text())
    renderer = RendererSpec.from_dict(job["renderer"])

    current_mesh_path: str | None = None
    current_norm: list | None = None
    meshes: list[bpy.types.Object] = []
    camera: bpy.types.Object | None = None

    configure_render(renderer)

    for task in job.get("tasks", []):
        output_path = Path(task["output_path"])
        if output_path.exists() and output_path.stat().st_size > 0:
            print(f"PROGRESS {task['id']}", flush=True)
            continue

        mesh_path = str(task["mesh_path"])
        normalization = task.get("normalization")
        camera_data = task.get("camera")
        mode = task.get("mode", "render")
        envmap = LightSpec.from_dict(task["envmap"])

        # Reload asset only if changed
        if mesh_path != current_mesh_path or normalization != current_norm:
            clear_scene()
            imported, meshes = import_asset(Path(mesh_path))
            if normalization:
                apply_normalization(imported, normalization)
            else:
                normalize_asset(imported, meshes)
            camera = add_camera(camera_data)
            current_mesh_path = mesh_path
            current_norm = normalization
        elif camera is not None:
            set_camera(camera, camera_data)

        setup_environment(envmap)

        if mode == "bake" and meshes:
            bake_combined_texture(meshes[0], output_path, renderer.resolution)
        else:
            render_png(output_path)

        print(f"PROGRESS {task['id']}", flush=True)


if __name__ == "__main__":
    main()
