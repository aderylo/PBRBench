"""Select a deterministic Objaverse subset with usable textured PBR materials."""

from __future__ import annotations

import argparse
import json
import random
import struct
from pathlib import Path
from typing import Any

import rootutils
import yaml

PROJECT_ROOT = rootutils.setup_root(
    __file__, indicator=".project_root", pythonpath=True
)

DEFAULT_INPUT = Path("/cluster/scratch/xiwang1/hiwi/PBRAgents/data/objaverse")
DEFAULT_OUTPUT = PROJECT_ROOT / "configs/data/splits/objaverse_pbr_64.yaml"


def read_glb_json(path: Path) -> dict[str, Any]:
    """Read only the JSON chunk, avoiding the usually much larger binary chunk."""
    with path.open("rb") as handle:
        header = handle.read(12)
        chunk_header = handle.read(8)
        if len(header) != 12 or len(chunk_header) != 8:
            raise ValueError("truncated GLB header")
        magic, version, _ = struct.unpack("<4sII", header)
        chunk_length, chunk_type = struct.unpack("<II", chunk_header)
        if magic != b"glTF" or version != 2 or chunk_type != 0x4E4F534A:
            raise ValueError("expected a GLB v2 JSON chunk")
        payload = handle.read(chunk_length)
    return json.loads(payload.rstrip(b" \t\r\n\x00"))


def has_textured_pbr_material(gltf: dict[str, Any]) -> bool:
    """Accept simple assets whose complete surface has one reusable PBR texture set."""
    materials = gltf.get("materials", [])
    if len(materials) != 1:
        return False

    def valid_uv0_texture(texture_info: Any) -> bool:
        if not isinstance(texture_info, dict) or texture_info.get("texCoord", 0) != 0:
            return False
        texture_index = texture_info.get("index")
        textures = gltf.get("textures", [])
        if not isinstance(texture_index, int) or not 0 <= texture_index < len(textures):
            return False
        image_index = textures[texture_index].get("source")
        return isinstance(image_index, int) and 0 <= image_index < len(
            gltf.get("images", [])
        )

    pbr = materials[0].get("pbrMetallicRoughness", {})
    if not valid_uv0_texture(pbr.get("baseColorTexture")):
        return False
    if not valid_uv0_texture(pbr.get("metallicRoughnessTexture")):
        return False

    primitives = [
        primitive
        for mesh in gltf.get("meshes", [])
        for primitive in mesh.get("primitives", [])
    ]
    return bool(primitives) and all(
        "TEXCOORD_0" in primitive.get("attributes", {})
        and primitive.get("material") == 0
        for primitive in primitives
    )


def scan(root: Path) -> tuple[list[str], int]:
    candidates = []
    invalid = 0
    for path in sorted((root / "glbs").rglob("*.glb")):
        try:
            if has_textured_pbr_material(read_glb_json(path)):
                candidates.append(path.stem)
        except (OSError, ValueError, json.JSONDecodeError, struct.error):
            invalid += 1
    return candidates, invalid


def make_split(candidates: list[str], count: int, seed: int) -> dict[str, Any]:
    if len(candidates) < count:
        raise ValueError(f"Requested {count} objects, but only found {len(candidates)}")

    selected = random.Random(seed).sample(sorted(candidates), count)
    train_count = count - 2 * max(1, count // 8)
    val_count = max(1, count // 8)
    records = [{"id": object_id} for object_id in selected]
    return {
        "dataset": "objaverse",
        "root": "data/objaverse",
        "repo_id": "allenai/objaverse",
        "path_template": "glbs/000-001/{id}.glb",
        "selection": {
            "seed": seed,
            "candidate_count": len(candidates),
            "criteria": [
                "one glTF material",
                "TEXCOORD_0 on every mesh primitive",
                "baseColorTexture",
                "metallicRoughnessTexture",
            ],
        },
        "train": records[:train_count],
        "val": records[train_count : train_count + val_count],
        "test": records[train_count + val_count :],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--count", type=int, default=64)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    arguments = parse_args()
    candidates, invalid = scan(arguments.input.expanduser().resolve())
    split = make_split(candidates, arguments.count, arguments.seed)
    output = arguments.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(yaml.safe_dump(split, sort_keys=False))
    print(
        f"Selected {arguments.count} of {len(candidates)} PBR objects "
        f"({invalid} invalid GLBs) and wrote {output}"
    )


if __name__ == "__main__":
    main()
