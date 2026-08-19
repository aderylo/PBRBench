"""Blender relighting execution and job orchestration for 3D PBR predictions."""

from __future__ import annotations

from dataclasses import asdict, dataclass, is_dataclass
import json
from pathlib import Path
import subprocess
import threading
from collections.abc import Sequence
from typing import Any, Literal

from tqdm.auto import tqdm

from src.utils import get_pylogger

log = get_pylogger(__name__)


@dataclass(frozen=True)
class RenderItem3D:
    """A single 3D relighting task (view render or texture bake)."""

    item_id: str
    mesh_path: Path
    normalization: list[list[float]] | None
    camera: dict[str, Any] | None
    envmap: Any
    output_path: Path
    mode: Literal["render", "bake"] = "render"

    def to_dict(self) -> dict[str, Any]:
        """Convert RenderItem3D to a JSON-serializable dictionary for Blender worker."""
        camera = (
            asdict(self.camera)
            if is_dataclass(self.camera)
            else (dict(self.camera) if self.camera is not None else None)
        )
        envmap_dict = (
            self.envmap.to_dict()
            if hasattr(self.envmap, "to_dict")
            else dict(self.envmap)
        )
        return {
            "id": self.item_id,
            "mesh_path": str(self.mesh_path.resolve()),
            "normalization": self.normalization,
            "camera": camera,
            "envmap": envmap_dict,
            "output_path": str(self.output_path.resolve()),
            "mode": self.mode,
        }


class Rerenderer3D:
    """Orchestrates Blender relighting and texture baking for 3D PBR assets."""

    def __init__(
        self,
        executable: str = "blender",
        resolution: int = 512,
        samples_per_pixel: int = 32,
        denoise: bool = True,
        device: str = "cuda",
        transparent_background: bool = True,
        workers: int = 1,
    ) -> None:
        self.executable = str(executable)
        self.resolution = int(resolution)
        self.samples_per_pixel = int(samples_per_pixel)
        self.denoise = bool(denoise)
        self.device = str(device)
        self.transparent_background = bool(transparent_background)
        self.workers = max(1, int(workers))
        self.helper = Path(__file__).parent / "rerender_3d_blender_worker.py"

    @property
    def renderer_spec(self) -> dict[str, Any]:
        """Convert rendering settings to a dictionary for the Blender worker."""
        return {
            "executable": self.executable,
            "resolution": self.resolution,
            "samples_per_pixel": self.samples_per_pixel,
            "denoise": self.denoise,
            "device": self.device,
            "transparent_background": self.transparent_background,
        }

    def render(
        self,
        items: Sequence[RenderItem3D],
        working_dir: Path | str,
        blender_log_path: Path | str,
        desc: str = "Indirect 3D PBR relighting (Blender)",
    ) -> None:
        """Run Blender 3D relighting subprocess(es) with live progress tracking."""
        if not items:
            log.warning("No render items to relight.")
            return

        working_dir = Path(working_dir).resolve()
        blender_log_path = Path(blender_log_path).resolve()
        log.info(f"Relighting {len(items)} 3D render tasks in Blender ({desc})")
        working_dir.mkdir(parents=True, exist_ok=True)
        serialized_tasks = [item.to_dict() for item in items]
        renderer_spec = self.renderer_spec

        num_workers = max(1, min(self.workers, len(items)))

        if num_workers == 1:
            job_spec = {
                "renderer": renderer_spec,
                "tasks": serialized_tasks,
            }
            job_path = working_dir / "job_3d.json"
            job_path.write_text(json.dumps(job_spec, indent=2))
            log.info(f"Blender log saved to {blender_log_path}")

            cmd = [
                self.executable,
                "--background",
                "--python",
                str(self.helper),
                "--",
                "--job",
                str(job_path),
            ]
            with open(blender_log_path, "a") as log_file:
                process = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1,
                )
                with tqdm(
                    total=len(items),
                    desc=desc,
                    unit="render",
                ) as pbar:
                    if process.stdout:
                        for line in process.stdout:
                            log_file.write(line)
                            if line.startswith("PROGRESS "):
                                pbar.update(1)
                return_code = process.wait()
                if return_code != 0:
                    raise subprocess.CalledProcessError(return_code, cmd)
            return

        # Multi-worker parallel execution
        log.info(f"Launching {num_workers} parallel Blender workers for 3D relighting")
        processes: list[subprocess.Popen] = []
        threads: list[threading.Thread] = []
        lock = threading.Lock()
        errors: list[Exception] = []

        with tqdm(
            total=len(items),
            desc=f"{desc} ({num_workers} Blender workers)",
            unit="render",
        ) as pbar:

            def stream_worker(
                proc: subprocess.Popen,
                log_path: Path,
                worker_id: int,
            ) -> None:
                try:
                    with open(log_path, "w") as log_file:
                        if proc.stdout:
                            for line in proc.stdout:
                                log_file.write(line)
                                if line.startswith("PROGRESS "):
                                    with lock:
                                        pbar.update(1)
                    rc = proc.wait()
                    if rc != 0:
                        with lock:
                            errors.append(
                                RuntimeError(
                                    f"Worker {worker_id} failed with exit code {rc}. See {log_path}"
                                )
                            )
                except Exception as e:
                    with lock:
                        errors.append(e)

            for worker_id in range(num_workers):
                worker_tasks = serialized_tasks[worker_id::num_workers]
                worker_job_spec = {
                    "renderer": renderer_spec,
                    "tasks": worker_tasks,
                }
                job_path = working_dir / f"job_3d_worker_{worker_id}.json"
                job_path.write_text(json.dumps(worker_job_spec, indent=2))
                worker_log_path = working_dir / f"blender_3d_worker_{worker_id}.log"

                cmd = [
                    self.executable,
                    "--background",
                    "--python",
                    str(self.helper),
                    "--",
                    "--job",
                    str(job_path),
                ]
                proc = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1,
                )
                processes.append(proc)
                t = threading.Thread(
                    target=stream_worker,
                    args=(proc, worker_log_path, worker_id),
                    daemon=True,
                )
                threads.append(t)
                t.start()

            for t in threads:
                t.join()

        # Append worker logs to combined log file
        with open(blender_log_path, "a") as main_log:
            for worker_id in range(num_workers):
                worker_log_path = working_dir / f"blender_3d_worker_{worker_id}.log"
                main_log.write(f"=== Worker {worker_id} Log ===\n")
                if worker_log_path.exists():
                    main_log.write(worker_log_path.read_text())
                    main_log.write("\n")

        if errors:
            for err in errors:
                log.error(f"Blender 3D worker error: {err}")
            for worker_id in range(num_workers):
                worker_log_path = working_dir / f"blender_3d_worker_{worker_id}.log"
                if worker_log_path.exists():
                    lines = worker_log_path.read_text().splitlines()
                    if lines:
                        log.error(f"--- Last 20 lines of Worker {worker_id} log ---")
                        for line in lines[-20:]:
                            log.error(f"  [Worker {worker_id}] {line}")
            raise errors[0]
