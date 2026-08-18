"""Patch TRELLIS 2 upstream texturing pipeline to preserve mesh UVs and handle writable geometry arrays.

1. By default, TRELLIS 2's ``preprocess_mesh`` instantiates a new ``trimesh.Trimesh``
   without passing ``visual=mesh.visual``. This drops the UV map and forces
   ``postprocess_mesh`` to fall back into slow CPU-bound ``cumesh.uv_unwrap()``
   (xatlas), taking 40+ minutes per sample on dense meshes.
   This script patches ``preprocess_mesh`` to pass ``visual=mesh.visual``.

2. When UVs are preserved and the fast path is taken, ``mesh.vertices`` and
   ``mesh.vertex_normals`` remain read-only ``TrackedArray`` views from ``trimesh``.
   This causes ``postprocess_mesh`` to fail at in-place coordinate swapping
   (``normals[:, 1], normals[:, 2] = normals[:, 2], -normals[:, 1]``) with:
   ``ValueError: assignment destination is read-only``.
   This script patches ``postprocess_mesh`` to create writable copies.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
TARGET_FILE = (
    PROJECT_ROOT
    / "third_party"
    / "TRELLIS.2"
    / "trellis2"
    / "pipelines"
    / "trellis2_texturing.py"
)

PATCH_1_ORIGINAL = "        return trimesh.Trimesh(vertices=vertices, faces=mesh.faces, process=False)"
PATCH_1_PATCHED = "        return trimesh.Trimesh(vertices=vertices, faces=mesh.faces, visual=mesh.visual, process=False)"

PATCH_2_ORIGINAL = """        vertices = mesh.vertices
        faces = mesh.faces
        normals = mesh.vertex_normals"""

PATCH_2_PATCHED = """        vertices = np.array(mesh.vertices, copy=True)
        faces = np.array(mesh.faces, copy=True)
        normals = np.array(mesh.vertex_normals, copy=True)"""

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def revert_trellis2_patch(file_path: Path = TARGET_FILE) -> bool:
    """Revert patches to restore original upstream TRELLIS 2 code."""
    if not file_path.is_file():
        raise FileNotFoundError(f"TRELLIS 2 texturing pipeline not found at: {file_path}")

    content = file_path.read_text(encoding="utf-8")
    reverted = False

    if PATCH_1_PATCHED in content:
        content = content.replace(PATCH_1_PATCHED, PATCH_1_ORIGINAL, 1)
        reverted = True

    if PATCH_2_PATCHED in content:
        content = content.replace(PATCH_2_PATCHED, PATCH_2_ORIGINAL, 1)
        reverted = True

    if reverted:
        file_path.write_text(content, encoding="utf-8")
        logger.info("Successfully reverted TRELLIS 2 texturing pipeline to original upstream code.")
    else:
        logger.info("TRELLIS 2 texturing pipeline is already in original unpatched state.")

    return True


def apply_trellis2_patch(file_path: Path = TARGET_FILE) -> bool:
    """Apply UV preservation and writable array patches to TRELLIS 2 texturing pipeline."""
    if not file_path.is_file():
        raise FileNotFoundError(f"TRELLIS 2 texturing pipeline not found at: {file_path}")

    content = file_path.read_text(encoding="utf-8")

    # Check / apply Patch 1 (UV preservation)
    if PATCH_1_PATCHED not in content:
        if PATCH_1_ORIGINAL not in content:
            raise RuntimeError(
                f"Failed to apply patch 1 to {file_path}:\n"
                f"Expected line not found:\n  {PATCH_1_ORIGINAL}\n"
                "Upstream TRELLIS 2 code structure may have changed."
            )
        content = content.replace(PATCH_1_ORIGINAL, PATCH_1_PATCHED, 1)
        logger.info("Applied Patch 1: Preserve mesh.visual in preprocess_mesh.")
    else:
        logger.info("Patch 1 (UV preservation) already present.")

    # Check / apply Patch 2 (Writable numpy arrays for normals/vertices)
    if PATCH_2_PATCHED not in content:
        if PATCH_2_ORIGINAL not in content:
            raise RuntimeError(
                f"Failed to apply patch 2 to {file_path}:\n"
                f"Expected lines not found:\n{PATCH_2_ORIGINAL}\n"
                "Upstream TRELLIS 2 code structure may have changed."
            )
        content = content.replace(PATCH_2_ORIGINAL, PATCH_2_PATCHED, 1)
        logger.info("Applied Patch 2: Make vertices and normals writable copies in postprocess_mesh.")
    else:
        logger.info("Patch 2 (Writable arrays) already present.")

    file_path.write_text(content, encoding="utf-8")
    logger.info("Successfully patched TRELLIS 2 texturing pipeline.")
    return True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--file",
        type=Path,
        default=TARGET_FILE,
        help="Path to trellis2_texturing.py",
    )
    parser.add_argument(
        "--revert",
        action="store_true",
        help="Revert the patch and restore original code",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.revert:
        revert_trellis2_patch(args.file)
    else:
        apply_trellis2_patch(args.file)
