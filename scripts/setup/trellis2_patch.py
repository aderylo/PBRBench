"""Patch TRELLIS 2 upstream texturing pipeline to preserve mesh UVs.

By default, TRELLIS 2's ``preprocess_mesh`` instantiates a new ``trimesh.Trimesh``
without passing ``visual=mesh.visual``. This drops the UV map and forces
``postprocess_mesh`` to fall back into slow CPU-bound ``cumesh.uv_unwrap()``
(xatlas), taking 40+ minutes per sample on dense meshes.

This script deliberately patches ``preprocess_mesh`` to pass ``visual=mesh.visual``.
If upstream updates and the code structure changes, this script fails loudly.
"""

from __future__ import annotations

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

TARGET_STR = "        return trimesh.Trimesh(vertices=vertices, faces=mesh.faces, process=False)"
PATCHED_STR = "        return trimesh.Trimesh(vertices=vertices, faces=mesh.faces, visual=mesh.visual, process=False)"

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def apply_trellis2_patch(file_path: Path = TARGET_FILE) -> bool:
    """Apply UV preservation patch to TRELLIS 2 texturing pipeline."""
    if not file_path.is_file():
        raise FileNotFoundError(f"TRELLIS 2 texturing pipeline not found at: {file_path}")

    content = file_path.read_text(encoding="utf-8")

    if PATCHED_STR in content:
        logger.info("TRELLIS 2 texturing pipeline is already patched (UVs preserved).")
        return True

    if TARGET_STR not in content:
        raise RuntimeError(
            f"Failed to patch {file_path}:\n"
            f"Expected line not found:\n  {TARGET_STR}\n"
            "Upstream TRELLIS 2 code structure may have changed. Please inspect the file manually."
        )

    patched_content = content.replace(TARGET_STR, PATCHED_STR, 1)
    file_path.write_text(patched_content, encoding="utf-8")
    logger.info("Successfully patched TRELLIS 2 texturing pipeline to preserve mesh UVs.")
    return True


if __name__ == "__main__":
    apply_trellis2_patch()
