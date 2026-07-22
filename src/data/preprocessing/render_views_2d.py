"""Render validated screen-space PBR observations from textured 3D assets."""

from __future__ import annotations

import subprocess
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path

import hydra
import rootutils
from omegaconf import DictConfig
from tqdm.auto import tqdm

rootutils.setup_root(__file__, indicator=".project_root", pythonpath=True)

from src.data.preprocessing.utils import (  # noqa: E402
    CameraSpec,
    LightSpec,
    RenderJob,
    RendererSpec,
    ViewSpec,
    asset_path,
    resolve_lights,
    resolved,
    selected_objects,
    orbit_views,
    write_json,
)


@dataclass(frozen=True)
class RenderFailure:
    """One object that could not be prepared by its Blender worker."""

    object_id: str
    return_code: int | None
    log_path: str
    error: str | None = None


def render_one(
    object_id: str, command: list[str], log_path: Path
) -> RenderFailure | None:
    """Run one Blender worker with its noisy output redirected to a log file."""
    try:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("w") as blender_output:
            completed = subprocess.run(
                command,
                check=False,
                stdout=blender_output,
                stderr=subprocess.STDOUT,
            )
    except OSError as error:
        return RenderFailure(object_id, None, str(log_path), str(error))
    if completed.returncode:
        return RenderFailure(object_id, completed.returncode, str(log_path))
    return None


def is_complete(job: RenderJob) -> bool:
    """Check whether every expected 2D artifact already exists for an object."""
    output_dir = Path(job.output_dir)
    for view in job.views:
        view_dir = output_dir / view.id
        expected = [view_dir / "metadata.json"]
        expected.extend(view_dir / "rgb" / f"{light.id}.png" for light in job.lights)
        expected.extend(
            view_dir / f"{name}.png"
            for name in ("albedo", "roughness", "metallic", "normal", "depth", "mask")
        )
        if not all(path.is_file() for path in expected):
            return False
    return True


def build_job(config: DictConfig, object_id: str, source_asset: Path) -> RenderJob:
    """Create the typed Blender payload for one object."""
    output_dir = Path(config.output_root) / object_id
    return RenderJob(
        object_id=object_id,
        asset_path=str(source_asset.resolve()),
        output_dir=str(output_dir.resolve()),
        views=tuple(ViewSpec(**view) for view in orbit_views(config)),
        camera=CameraSpec.from_dict(resolved(config.views)),
        lights=tuple(LightSpec.from_dict(light) for light in resolve_lights(config)),
        renderer=RendererSpec.from_dict(resolved(config.rendering)),
        overwrite=not bool(config.resume),
        min_foreground_pixels=int(config.min_foreground_pixels),
    )


@hydra.main(
    version_base=None,
    config_path="../../../configs",
    config_name="data/preprocessing/texverse_2d",
)
def main(config: DictConfig) -> None:
    object_ids = selected_objects(config)
    helper = Path(__file__).with_name("_render_views_2d.py")
    failures: list[RenderFailure] = []
    log_dir = Path(config.blender_log_dir)
    with tempfile.TemporaryDirectory(prefix="pbr_render_2d_") as temporary_directory:
        jobs_dir = Path(temporary_directory)
        for object_id in tqdm(object_ids, desc="Rendering 2D", unit="object"):
            job = build_job(config, object_id, asset_path(config, object_id))
            if bool(config.resume) and is_complete(job):
                continue
            job_path = jobs_dir / f"{object_id}.json"
            write_json(job_path, job.to_dict())
            command = [
                str(config.rendering.executable),
                "--background",
                "--python",
                str(helper),
                "--",
                "--job",
                str(job_path),
            ]
            log_path = log_dir / f"{object_id}.log"
            if config.dry_run:
                print(" ".join(command))
                continue
            failure = render_one(object_id, command, log_path)
            if failure is not None:
                failures.append(failure)

    if failures:
        report_path = Path(config.output_root) / "render_2d_failures.json"
        write_json(report_path, {"failures": [asdict(item) for item in failures]})
        raise RuntimeError(
            f"{len(failures)} of {len(object_ids)} objects failed preparation; "
            f"see {report_path}"
        )


if __name__ == "__main__":
    main()
