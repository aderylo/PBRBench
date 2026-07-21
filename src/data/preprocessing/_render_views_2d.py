"""Blender-side implementation for registered 2D evaluation views."""

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
    return imported, meshes


def bounds(meshes: list[bpy.types.Object]) -> tuple[Vector, float]:
    bpy.context.view_layer.update()
    points = [
        item.matrix_world @ Vector(corner)
        for item in meshes
        for corner in item.bound_box
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


def add_camera(camera_config: dict) -> bpy.types.Object:
    bpy.ops.object.camera_add()
    camera = bpy.context.object
    camera.name = "BenchmarkCamera"
    camera.data.lens = float(camera_config["focal_length_mm"])
    camera.data.sensor_width = float(camera_config["sensor_width_mm"])
    camera.data.sensor_fit = "HORIZONTAL"
    bpy.context.scene.camera = camera
    return camera


def place_camera(
    camera: bpy.types.Object, yaw_deg: float, elevation_deg: float, distance: float
) -> None:
    yaw = math.radians(yaw_deg)
    elevation = math.radians(elevation_deg)
    camera.location = (
        distance * math.cos(elevation) * math.cos(yaw),
        distance * math.cos(elevation) * math.sin(yaw),
        distance * math.sin(elevation),
    )
    camera.rotation_euler = (-camera.location).to_track_quat("-Z", "Y").to_euler()
    bpy.context.view_layer.update()


def configure_render(config: dict) -> None:
    scene = bpy.context.scene
    resolution = int(config["resolution"])
    scene.render.resolution_x = resolution
    scene.render.resolution_y = resolution
    scene.render.resolution_percentage = 100
    scene.render.film_transparent = bool(config.get("transparent_background", True))
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGBA"
    scene.render.image_settings.color_depth = "8"
    scene.render.engine = "CYCLES"
    scene.cycles.samples = int(config["samples_per_pixel"])
    scene.cycles.use_denoising = bool(config.get("denoise", True))
    scene.cycles.seed = 0
    scene.view_settings.view_transform = "Standard"
    scene.view_settings.look = "None"
    scene.view_settings.exposure = 0.0
    scene.view_settings.gamma = 1.0

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


def setup_environment(item: dict) -> None:
    world = bpy.context.scene.world or bpy.data.worlds.new("BenchmarkWorld")
    bpy.context.scene.world = world
    world.use_nodes = True
    nodes = world.node_tree.nodes
    links = world.node_tree.links
    nodes.clear()
    texture = nodes.new(type="ShaderNodeTexEnvironment")
    texture.image = bpy.data.images.load(item["path"], check_existing=True)
    mapping = nodes.new(type="ShaderNodeMapping")
    mapping.inputs["Rotation"].default_value[2] = math.radians(
        float(item["rotation_deg"])
    )
    coordinates = nodes.new(type="ShaderNodeTexCoord")
    background = nodes.new(type="ShaderNodeBackground")
    background.inputs["Strength"].default_value = float(item["strength"])
    output = nodes.new(type="ShaderNodeOutputWorld")
    links.new(coordinates.outputs["Generated"], mapping.inputs["Vector"])
    links.new(mapping.outputs["Vector"], texture.inputs["Vector"])
    links.new(texture.outputs["Color"], background.inputs["Color"])
    links.new(background.outputs["Background"], output.inputs["Surface"])


def render_png(
    path: Path,
    *,
    color_mode: str = "RGBA",
    color_depth: str = "8",
    transform: str = "Raw",
) -> None:
    scene = bpy.context.scene
    path.parent.mkdir(parents=True, exist_ok=True)
    scene.render.image_settings.color_mode = color_mode
    scene.render.image_settings.color_depth = color_depth
    scene.view_settings.view_transform = transform
    scene.render.filepath = str(path)
    bpy.ops.render.render(write_still=True)


def all_materials(meshes: list[bpy.types.Object]) -> list[bpy.types.Material]:
    materials = []
    seen = set()
    for mesh in meshes:
        for material in mesh.data.materials:
            if material and material.name_full not in seen:
                material.use_nodes = True
                materials.append(material)
                seen.add(material.name_full)
    return materials


def channel_source(material: bpy.types.Material, channel: str):
    nodes = material.node_tree.nodes
    principled = next((node for node in nodes if node.type == "BSDF_PRINCIPLED"), None)
    if principled is None:
        return None, None
    socket_name = {
        "albedo": "Base Color",
        "roughness": "Roughness",
        "metallic": "Metallic",
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
        fallback = (
            material.diffuse_color
            if channel == "albedo"
            else 1.0
            if channel == "roughness"
            else 0.0
        )
        if hasattr(fallback, "__len__"):
            emission.inputs["Color"].default_value = tuple(fallback)
        else:
            emission.inputs["Color"].default_value = (fallback, fallback, fallback, 1.0)
    links.new(emission.outputs["Emission"], output.inputs["Surface"])


def emission_material(
    name: str, kind: str, near: float = 0.0, far: float = 1.0
) -> bpy.types.Material:
    material = bpy.data.materials.new(name)
    material.use_nodes = True
    nodes = material.node_tree.nodes
    links = material.node_tree.links
    nodes.clear()
    emission = nodes.new(type="ShaderNodeEmission")
    output = nodes.new(type="ShaderNodeOutputMaterial")

    if kind == "mask":
        emission.inputs["Color"].default_value = (1.0, 1.0, 1.0, 1.0)
    elif kind == "normal":
        geometry = nodes.new(type="ShaderNodeNewGeometry")
        transform = nodes.new(type="ShaderNodeVectorTransform")
        transform.vector_type = "NORMAL"
        transform.convert_from = "WORLD"
        transform.convert_to = "CAMERA"
        scale = nodes.new(type="ShaderNodeVectorMath")
        scale.operation = "SCALE"
        scale.inputs["Scale"].default_value = 0.5
        add = nodes.new(type="ShaderNodeVectorMath")
        add.operation = "ADD"
        add.inputs[1].default_value = (0.5, 0.5, 0.5)
        links.new(geometry.outputs["Normal"], transform.inputs["Vector"])
        links.new(transform.outputs["Vector"], scale.inputs[0])
        links.new(scale.outputs["Vector"], add.inputs[0])
        links.new(add.outputs["Vector"], emission.inputs["Color"])
    elif kind == "depth":
        camera_data = nodes.new(type="ShaderNodeCameraData")
        mapping = nodes.new(type="ShaderNodeMapRange")
        mapping.inputs["From Min"].default_value = near
        mapping.inputs["From Max"].default_value = far
        mapping.inputs["To Min"].default_value = 0.0
        mapping.inputs["To Max"].default_value = 1.0
        if hasattr(mapping, "clamp"):
            mapping.clamp = True
        if hasattr(mapping, "use_clamp"):
            mapping.use_clamp = True
        links.new(camera_data.outputs["View Z Depth"], mapping.inputs["Value"])
        links.new(mapping.outputs["Result"], emission.inputs["Color"])
    links.new(emission.outputs["Emission"], output.inputs["Surface"])
    return material


def depth_range(
    camera: bpy.types.Object, meshes: list[bpy.types.Object]
) -> tuple[float, float]:
    world_to_camera = camera.matrix_world.inverted()
    depths = [
        -(world_to_camera @ (mesh.matrix_world @ Vector(corner))).z
        for mesh in meshes
        for corner in mesh.bound_box
    ]
    near, far = min(depths), max(depths)
    return float(near), float(max(far, near + 1e-6))


def render_reference_passes(
    view_dir: Path, meshes: list[bpy.types.Object], camera: bpy.types.Object
) -> tuple[float, float]:
    scene = bpy.context.scene
    scene.render.engine = "BLENDER_EEVEE"
    materials = all_materials(meshes)
    for channel in ("albedo", "roughness", "metallic"):
        for material in materials:
            set_material_channel(material, channel)
        transform = "Standard" if channel == "albedo" else "Raw"
        mode = "RGBA" if channel == "albedo" else "BW"
        render_png(
            view_dir / f"{channel}.png",
            color_mode=mode,
            color_depth="8",
            transform=transform,
        )

    near, far = depth_range(camera, meshes)
    for channel, material, mode, bits in (
        ("normal", emission_material("BenchmarkNormal", "normal"), "RGB", "16"),
        ("depth", emission_material("BenchmarkDepth", "depth", near, far), "BW", "16"),
        ("mask", emission_material("BenchmarkMask", "mask"), "BW", "8"),
    ):
        bpy.context.view_layer.material_override = material
        render_png(
            view_dir / f"{channel}.png",
            color_mode=mode,
            color_depth=bits,
            transform="Raw",
        )
    bpy.context.view_layer.material_override = None
    return near, far


def camera_metadata(camera: bpy.types.Object, resolution: int) -> dict:
    focal = float(camera.data.lens)
    sensor = float(camera.data.sensor_width)
    f_pixels = focal / sensor * resolution
    return {
        "model": "perspective",
        "resolution": [resolution, resolution],
        "intrinsics": [
            [f_pixels, 0.0, resolution / 2],
            [0.0, f_pixels, resolution / 2],
            [0.0, 0.0, 1.0],
        ],
        "camera_to_world": matrix_rows(camera.matrix_world),
        "world_to_camera": matrix_rows(camera.matrix_world.inverted()),
        "coordinate_system": "Blender camera: +X right, +Y up, -Z forward",
    }


def main() -> None:
    job = json.loads(arguments().job.read_text())
    output_dir = Path(job["output_dir"])
    clear_scene()
    imported, meshes = import_asset(Path(job["asset_path"]))
    normalization = normalize_asset(imported, meshes)
    camera = add_camera(job["camera"])
    configure_render(job["renderer"])
    original_engine = bpy.context.scene.render.engine

    # Render every illumination observation before modifying material graphs for
    # reference passes. Otherwise the next view would see the preceding PBR
    # channel material instead of the original authored material.
    pending_views = []
    for view in job["views"]:
        view_dir = output_dir / view["id"]
        metadata_path = view_dir / "metadata.json"
        expected = [view_dir / "rgb" / f"{item['id']}.png" for item in job["lights"]]
        expected += [
            view_dir / f"{name}.png"
            for name in ("albedo", "roughness", "metallic", "normal", "depth", "mask")
        ]
        if (
            metadata_path.is_file()
            and all(path.is_file() for path in expected)
            and not job["overwrite"]
        ):
            print(f"skip complete {job['object_id']}/{view['id']}")
            continue

        pending_views.append(view)

        place_camera(
            camera,
            view["yaw_deg"],
            float(job["camera"]["elevation_deg"]),
            float(job["camera"]["distance"]),
        )
        bpy.context.scene.render.engine = original_engine
        for light in job["lights"]:
            setup_environment(light)
            render_png(
                view_dir / "rgb" / f"{light['id']}.png",
                color_mode="RGBA",
                color_depth="8",
                transform="Standard",
            )

    for view in pending_views:
        view_dir = output_dir / view["id"]
        metadata_path = view_dir / "metadata.json"
        place_camera(
            camera,
            view["yaw_deg"],
            float(job["camera"]["elevation_deg"]),
            float(job["camera"]["distance"]),
        )
        near, far = render_reference_passes(view_dir, meshes, camera)
        metadata = {
            "schema_version": 2,
            "representation": "2d",
            "object_id": job["object_id"],
            "asset_path": job["asset_path"],
            "view_id": view["id"],
            "yaw_deg": view["yaw_deg"],
            "elevation_deg": job["camera"]["elevation_deg"],
            "camera": camera_metadata(camera, int(job["renderer"]["resolution"])),
            "normalization_source_to_world": matrix_rows(normalization),
            "normal": {"space": "camera", "encoding": "uint16 PNG, (normal + 1) / 2"},
            "depth": {
                "encoding": "uint16 PNG",
                "near": near,
                "far": far,
                "value": "(z - near) / (far - near)",
            },
            "lights": [
                {
                    "id": item["id"],
                    "rotation_deg": item["rotation_deg"],
                    "strength": item["strength"],
                    "rgb": f"rgb/{item['id']}.png",
                }
                for item in job["lights"]
            ],
            "renderer": job["renderer"],
            "blender_version": bpy.app.version_string,
        }
        metadata_path.parent.mkdir(parents=True, exist_ok=True)
        metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")


if __name__ == "__main__":
    main()
