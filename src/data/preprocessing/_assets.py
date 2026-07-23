"""Shared canonical source-asset storage and glTF-to-GLB conversion helpers."""

from __future__ import annotations

import base64
import json
import mimetypes
import os
import shutil
import struct
import tempfile
import urllib.parse
from pathlib import Path, PurePosixPath
from typing import Any

ASSET_DIRECTORY_NAME = "assets"


def safe_component(value: str, label: str) -> str:
    """Validate a value used as exactly one local path component."""
    path = Path(value)
    if not value or path.name != value or value in {".", ".."}:
        raise ValueError(f"{label} is not a safe path component: {value!r}")
    return value


def canonical_asset_path(assets_root: Path, source: str, object_id: str) -> Path:
    """Return ``<assets_root>/<source>/<object_id>.glb`` safely."""
    return assets_root / safe_component(source, "Source name") / (
        safe_component(object_id, "Object ID") + ".glb"
    )


def install_file(source: Path, destination: Path) -> None:
    """Atomically copy a source GLB into its canonical destination."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=destination.parent, prefix=f".{destination.name}.", suffix=".part", delete=False
    ) as temporary:
        temporary_path = Path(temporary.name)
        with source.open("rb") as input_file:
            shutil.copyfileobj(input_file, temporary)
    try:
        os.replace(temporary_path, destination)
    finally:
        temporary_path.unlink(missing_ok=True)


def _resource_path(base_dir: Path, uri: str) -> Path:
    path = PurePosixPath(uri)
    if not uri or path.is_absolute() or ".." in path.parts or "." in path.parts:
        raise ValueError(f"glTF contains an unsafe external resource URI: {uri!r}")
    return base_dir.joinpath(*path.parts)


def _decode_data_uri(uri: str) -> tuple[bytes, str | None]:
    try:
        header, encoded = uri.split(",", 1)
    except ValueError as error:
        raise ValueError("Malformed glTF data URI") from error
    if not header.startswith("data:"):
        raise ValueError(f"Unsupported glTF resource URI: {uri!r}")
    mime_type = header[5:].split(";", 1)[0] or None
    try:
        payload = (
            base64.b64decode(encoded, validate=True)
            if ";base64" in header
            else urllib.parse.unquote_to_bytes(encoded)
        )
    except (ValueError, UnicodeEncodeError) as error:
        raise ValueError("Malformed glTF data URI payload") from error
    return payload, mime_type


def _read_resource(base_dir: Path, uri: str) -> tuple[bytes, str | None]:
    if uri.startswith("data:"):
        return _decode_data_uri(uri)
    path = _resource_path(base_dir, uri)
    try:
        return path.read_bytes(), mimetypes.guess_type(path.name)[0]
    except OSError as error:
        raise ValueError(f"Missing glTF resource: {path}") from error


def _append_binary(binary: bytearray, payload: bytes) -> int:
    offset = len(binary)
    binary.extend(payload)
    binary.extend(b"\0" * ((-len(binary)) % 4))
    return offset


def gltf_to_glb(gltf_path: Path, output_path: Path) -> None:
    """Embed a glTF JSON asset and all local resources into one GLB file.

    Poly Haven publishes glTF JSON plus external buffers and textures.  The
    conversion keeps those resources self-contained, so all dataset sources
    share the canonical one-file GLB format.
    """
    try:
        document: dict[str, Any] = json.loads(gltf_path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"Invalid glTF JSON: {gltf_path}") from error

    buffers = document.get("buffers", [])
    buffer_views = document.get("bufferViews", [])
    if not isinstance(buffers, list) or not isinstance(buffer_views, list):
        raise ValueError(f"Malformed glTF buffer definitions: {gltf_path}")

    binary = bytearray()
    buffer_offsets: list[int] = []
    for index, buffer in enumerate(buffers):
        if not isinstance(buffer, dict):
            raise ValueError(f"Malformed glTF buffer {index}: {gltf_path}")
        uri = buffer.get("uri")
        expected_size = buffer.get("byteLength")
        if not isinstance(expected_size, int) or expected_size < 0:
            raise ValueError(f"Malformed glTF buffer byteLength {index}: {gltf_path}")
        if uri is None and expected_size == 0:
            payload = b""
        elif not isinstance(uri, str):
            raise ValueError(f"glTF buffer {index} has no readable URI: {gltf_path}")
        else:
            payload, _ = _read_resource(gltf_path.parent, uri)
        if len(payload) < expected_size:
            raise ValueError(f"glTF buffer {index} is shorter than declared: {gltf_path}")
        buffer_offsets.append(_append_binary(binary, payload))

    for index, view in enumerate(buffer_views):
        if not isinstance(view, dict) or not isinstance(view.get("buffer"), int):
            raise ValueError(f"Malformed glTF buffer view {index}: {gltf_path}")
        source_buffer = view["buffer"]
        if source_buffer < 0 or source_buffer >= len(buffer_offsets):
            raise ValueError(f"glTF buffer view {index} references an unknown buffer")
        view["byteOffset"] = buffer_offsets[source_buffer] + int(view.get("byteOffset", 0))
        view["buffer"] = 0

    images = document.get("images", [])
    if not isinstance(images, list):
        raise ValueError(f"Malformed glTF images: {gltf_path}")
    for index, image in enumerate(images):
        if not isinstance(image, dict) or "uri" not in image:
            continue
        uri = image.pop("uri")
        if not isinstance(uri, str):
            raise ValueError(f"Malformed glTF image {index}: {gltf_path}")
        payload, data_mime_type = _read_resource(gltf_path.parent, uri)
        mime_type = image.get("mimeType") or data_mime_type
        if not mime_type:
            raise ValueError(f"Could not determine MIME type for glTF image {uri!r}")
        image["mimeType"] = mime_type
        image["bufferView"] = len(buffer_views)
        buffer_views.append({"buffer": 0, "byteOffset": _append_binary(binary, payload), "byteLength": len(payload)})

    document["buffers"] = [{"byteLength": len(binary)}]
    document["bufferViews"] = buffer_views
    json_chunk = json.dumps(document, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    json_chunk += b" " * ((-len(json_chunk)) % 4)
    binary_chunk = bytes(binary)
    length = 12 + 8 + len(json_chunk) + 8 + len(binary_chunk)
    glb = b"".join(
        (
            struct.pack("<4sII", b"glTF", 2, length),
            struct.pack("<I4s", len(json_chunk), b"JSON"),
            json_chunk,
            struct.pack("<I4s", len(binary_chunk), b"BIN\0"),
            binary_chunk,
        )
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(glb)
