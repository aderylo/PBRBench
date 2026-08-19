"""Blender-side projective relighting for screen-space PBR maps."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import bpy  # noqa: E402
from mathutils import Matrix  # noqa: E402

from src.data.preprocessing._render_samples_2d import (  # noqa: E402
    clear_scene,
    configure_render,
    import_asset,
    render_png,
    setup_environment,
)
from src.data.preprocessing.utils import (  # noqa: E402
    LightSpec,
    RendererSpec,
)

UV_LAYER = "BenchmarkProjection"


def arguments() -> argparse.Namespace:
    argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    parser = argparse.ArgumentParser()
    parser.add_argument("--job", type=Path, required=True)
    return parser.parse_args(argv)


def apply_normalization(imported: list, rows: list[list[float]]) -> None:
    transform = Matrix(rows)
    imported_set = set(imported)
    for item in [item for item in imported if item.parent not in imported_set]:
        item.matrix_world = transform @ item.matrix_world
    bpy.context.view_layer.update()


def add_camera(camera_data: dict):
    bpy.ops.object.camera_add()
    camera = bpy.context.object
    camera.name = "BenchmarkCamera"
    width = int(camera_data["resolution"][0])
    camera.data.sensor_width = 36.0
    camera.data.sensor_height = 36.0
    camera.data.lens = float(camera_data["intrinsics"][0][0]) * 36.0 / width
    camera.matrix_world = Matrix(camera_data["camera_to_world"])
    bpy.context.scene.camera = camera
    bpy.context.view_layer.update()
    return camera


def set_camera(camera, camera_data: dict) -> None:
    camera.matrix_world = Matrix(camera_data["camera_to_world"])
    bpy.context.view_layer.update()


def add_projection(meshes: list, camera) -> None:
    for mesh in meshes:
        if mesh.data.uv_layers.get(UV_LAYER) is None:
            mesh.data.uv_layers.new(name=UV_LAYER)
        modifier = mesh.modifiers.new("BenchmarkUVProject", "UV_PROJECT")
        modifier.uv_layer = UV_LAYER
        modifier.projector_count = 1
        modifier.projectors[0].object = camera
        modifier.aspect_x = 1.0
        modifier.aspect_y = 1.0


def projected_material(paths: dict[str, str]):
    material = bpy.data.materials.new("BenchmarkProjectedPBR")
    material.use_nodes = True
    nodes = material.node_tree.nodes
    links = material.node_tree.links
    nodes.clear()
    output = nodes.new(type="ShaderNodeOutputMaterial")
    principled = nodes.new(type="ShaderNodeBsdfPrincipled")
    uv = nodes.new(type="ShaderNodeUVMap")
    uv.uv_map = UV_LAYER
    links.new(principled.outputs["BSDF"], output.inputs["Surface"])

    for channel, socket_name in (
        ("albedo", "Base Color"),
        ("roughness", "Roughness"),
        ("metallic", "Metallic"),
    ):
        texture = nodes.new(type="ShaderNodeTexImage")
        texture.image = bpy.data.images.load(paths[channel], check_existing=False)
        texture.image.colorspace_settings.name = (
            "sRGB" if channel == "albedo" else "Non-Color"
        )
        texture.interpolation = "Linear"
        texture.extension = "CLIP"
        links.new(uv.outputs["UV"], texture.inputs["Vector"])
        links.new(texture.outputs["Color"], principled.inputs[socket_name])
    return material


def cleanup_projected_materials() -> None:
    materials = [
        material
        for material in bpy.data.materials
        if material.name.startswith("BenchmarkProjectedPBR")
    ]
    images = {
        node.image
        for material in materials
        for node in material.node_tree.nodes
        if node.type == "TEX_IMAGE" and node.image is not None
    }
    for material in materials:
        bpy.data.materials.remove(material)
    for image in images:
        if image.users == 0:
            bpy.data.images.remove(image)


def assign_material(meshes: list, paths: dict[str, str]) -> None:
    for mesh in meshes:
        mesh.data.materials.clear()
    cleanup_projected_materials()
    material = projected_material(paths)
    for mesh in meshes:
        mesh.data.materials.append(material)


def main() -> None:
    job = json.loads(arguments().job.read_text())
    renderer = RendererSpec.from_dict(job["renderer"])

    current_asset: str | None = None
    current_norm: list | None = None
    meshes: list = []
    camera = None

    configure_render(renderer)

    for task in job.get("tasks", []):
        output_path = Path(task["output_path"])
        if output_path.exists() and output_path.stat().st_size > 0:
            print(f"PROGRESS {task['id']}", flush=True)
            continue

        asset_path = str(task["asset_path"])
        normalization = task["normalization"]

        # Only reload asset if different from current
        if asset_path != current_asset or normalization != current_norm:
            clear_scene()
            imported, meshes = import_asset(Path(asset_path))
            apply_normalization(imported, normalization)
            camera = add_camera(task["camera"])
            add_projection(meshes, camera)
            current_asset = asset_path
            current_norm = normalization
        else:
            set_camera(camera, task["camera"])

        assign_material(meshes, task["channels"])
        setup_environment(LightSpec.from_dict(task["envmap"]))
        render_png(output_path, transform="Standard")
        print(f"PROGRESS {task['id']}", flush=True)


if __name__ == "__main__":
    main()
