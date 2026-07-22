"""Bake one light-conditioned texture per normalized 3D evaluation object."""

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
    BakeJob,
    LightSpec,
    RendererSpec,
    asset_path,
    resolve_lights,
    resolved,
    selected_objects,
    write_json,
)


@dataclass(frozen=True)
class BakeFailure:
    """One object that could not be baked by its Blender worker."""

    object_id: str
    return_code: int | None
    log_path: str
    error: str | None = None


def render_one(
    object_id: str, command: list[str], log_path: Path
) -> BakeFailure | None:
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
        return BakeFailure(object_id, None, str(log_path), str(error))
    if completed.returncode:
        return BakeFailure(object_id, completed.returncode, str(log_path))
    return None


def is_complete(job: BakeJob) -> bool:
    """Check whether every expected 3D artifact already exists for an object."""
    output_dir = Path(job.output_dir)
    expected = [output_dir / "mesh.obj", output_dir / "metadata.json"]
    expected.extend(output_dir / "textures" / f"{light.id}.png" for light in job.lights)
    return all(path.is_file() for path in expected)


def build_job(config: DictConfig, object_id: str, source_asset: Path) -> BakeJob:
    """Create the typed Blender payload for one object-level 3D sample."""
    return BakeJob(
        object_id=object_id,
        asset_path=str(source_asset.resolve()),
        output_dir=str((Path(config.output_root) / object_id).resolve()),
        lights=tuple(LightSpec.from_dict(light) for light in resolve_lights(config)),
        renderer=RendererSpec.from_dict(resolved(config.rendering)),
        overwrite=not bool(config.resume),
    )


@hydra.main(
    version_base=None,
    config_path="../../../configs",
    config_name="data/preprocessing/texverse_3d",
)
def main(config: DictConfig) -> None:
    object_ids = selected_objects(config)
    helper = Path(__file__).with_name("_render_views_3d.py")
    failures: list[BakeFailure] = []
    log_dir = Path(config.blender_log_dir)
    with tempfile.TemporaryDirectory(prefix="pbr_render_3d_") as temporary_directory:
        jobs_dir = Path(temporary_directory)
        for object_id in tqdm(object_ids, desc="Rendering 3D", unit="object"):
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
        report_path = Path(config.output_root) / "render_3d_failures.json"
        write_json(report_path, {"failures": [asdict(item) for item in failures]})
        raise RuntimeError(
            f"{len(failures)} of {len(object_ids)} objects failed preparation; "
            f"see {report_path}"
        )


if __name__ == "__main__":
    main()
