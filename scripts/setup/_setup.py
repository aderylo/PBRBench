"""Shared utilities for preparing method-specific environments."""

from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

_REQUIREMENT_SPLIT = re.compile(r"[=<>!~\[;]")


def _requirement_name(line: str) -> str:
    """Extract the normalized package name from a requirements line, if any."""
    if not line or line.startswith(("#", "--", "-e")):
        return ""
    return _REQUIREMENT_SPLIT.split(line)[0].strip().lower()


def _normalize_legacy_editables(
    requirements: Path,
    *,
    drop_index_lines: bool = False,
    drop_requirements: tuple[str, ...] = (),
) -> tuple[Path, tuple[str, ...], Path | None]:
    """Convert legacy editable Git lines into uv-compatible direct references.

    When ``drop_index_lines`` is set, ``--index-url`` / ``--extra-index-url`` /
    ``--find-links`` directives are removed so that uv resolves everything
    from its configured default index. When ``drop_requirements`` is set,
    lines pinning those package names are removed as well (e.g. to override
    an upstream version pin from the command line).
    """
    kept_lines: list[str] = []
    direct_requirements: list[str] = []
    dropped_lines = False

    for line in requirements.read_text().splitlines(keepends=True):
        stripped = line.strip()
        if drop_index_lines and stripped.startswith(
            ("--index-url", "--extra-index-url", "--find-links")
        ):
            dropped_lines = True
        elif stripped.startswith("-e git+") and "#egg=" in stripped:
            package = stripped.split("#egg=", 1)[1].split("&", 1)[0]
            git_url = stripped[3:].split("#", 1)[0]
            direct_requirements.append(f"{package} @ {git_url}")
        elif _requirement_name(stripped) in drop_requirements:
            dropped_lines = True
        else:
            kept_lines.append(line)

    if not direct_requirements and not dropped_lines:
        return requirements, (), None

    with tempfile.NamedTemporaryFile(
        mode="w",
        prefix="pbr-eval-requirements-",
        suffix=".txt",
        delete=False,
    ) as temporary:
        temporary.write("".join(kept_lines))

    temporary_path = Path(temporary.name)
    return temporary_path, tuple(direct_requirements), temporary_path


def setup_method(
    method: str,
    repository: str,
    *,
    python: str = "3.11",
    requirements: str | None = None,
    bootstrap_requirements: tuple[str, ...] = (),
    exclude_requirements: str | None = None,
    extra_requirements: tuple[str, ...] = (),
    index_strategy: str | None = None,
    torch_backend: str | None = None,
    find_links: tuple[str, ...] = (),
    no_build_isolation_packages: tuple[str, ...] = (),
    drop_index_lines: bool = False,
    drop_requirements: tuple[str, ...] = (),
) -> None:
    """Create a uv environment containing the benchmark and method dependencies."""
    repository_root = PROJECT_ROOT / repository
    if requirements is None:
        requirements_path = repository_root / "requirements.txt"
    else:
        requirements_path = PROJECT_ROOT / requirements
    if not requirements_path.is_file():
        raise FileNotFoundError(requirements_path)

    environment = PROJECT_ROOT / "third_party/.venvs" / method
    subprocess.run(
        ["uv", "venv", "--python", python, "--allow-existing", str(environment)],
        check=True,
    )
    if bootstrap_requirements:
        subprocess.run(
            [
                "uv",
                "pip",
                "install",
                "--python",
                str(environment / "bin/python"),
                *bootstrap_requirements,
                *(
                    ("--index-strategy", index_strategy)
                    if index_strategy is not None
                    else ()
                ),
                *(
                    ("--torch-backend", torch_backend)
                    if torch_backend is not None
                    else ()
                ),
            ],
            check=True,
        )

    install_requirements, converted_editables, temporary_requirements = (
        _normalize_legacy_editables(
            requirements_path,
            drop_index_lines=drop_index_lines,
            drop_requirements=drop_requirements,
        )
    )
    if converted_editables:
        print(
            "Converted legacy editable Git requirements: "
            + ", ".join(converted_editables)
        )

    install_command = [
        "uv",
        "pip",
        "install",
        "--python",
        str(environment / "bin/python"),
        "-r",
        str(PROJECT_ROOT / "pyproject.toml"),
        "-r",
        str(install_requirements),
        *converted_editables,
        *extra_requirements,
        *(("--index-strategy", index_strategy) if index_strategy is not None else ()),
        *(("--torch-backend", torch_backend) if torch_backend is not None else ()),
        *(option for link in find_links for option in ("--find-links", link)),
        *(
            ("--excludes", str(PROJECT_ROOT / exclude_requirements))
            if exclude_requirements is not None
            else ()
        ),
        *(
            option
            for package in no_build_isolation_packages
            for option in ("--no-build-isolation-package", package)
        ),
    ]
    try:
        subprocess.run(install_command, check=True)
    finally:
        if temporary_requirements is not None:
            temporary_requirements.unlink(missing_ok=True)
    print(f"Ready: {environment / 'bin/python'}")


def setup_method_cli(
    method: str,
    repository: str,
    *,
    python_default: str = "3.11",
    requirements: str | None = None,
    bootstrap_requirements: tuple[str, ...] = (),
    exclude_requirements: str | None = None,
    extra_requirements: tuple[str, ...] = (),
    index_strategy: str | None = None,
    torch_backend: str | None = None,
    find_links: tuple[str, ...] = (),
    no_build_isolation_packages: tuple[str, ...] = (),
    drop_index_lines: bool = False,
    drop_requirements: tuple[str, ...] = (),
) -> None:
    """Parse common command-line options and prepare a method environment."""
    parser = argparse.ArgumentParser(
        description=f"Create the uv environment for {method}."
    )
    parser.add_argument("--python", default=python_default)
    args = parser.parse_args()
    setup_method(
        method,
        repository,
        python=args.python,
        requirements=requirements,
        bootstrap_requirements=bootstrap_requirements,
        exclude_requirements=exclude_requirements,
        extra_requirements=extra_requirements,
        index_strategy=index_strategy,
        torch_backend=torch_backend,
        find_links=find_links,
        no_build_isolation_packages=no_build_isolation_packages,
        drop_index_lines=drop_index_lines,
        drop_requirements=drop_requirements,
    )


def install_cuda_extensions(
    method: str,
    extensions: tuple[str, ...],
    *,
    label: str | None = None,
) -> None:
    """Install CUDA C++ extensions into a method environment.

    The extensions are compiled against the torch build already present in
    the environment (installed without build isolation). When no ``nvcc``
    compiler is available the step is skipped with a warning; on Euler,
    first load: ``module load stack/2024-06 gcc/12.2.0 cuda/12.4.1``.
    """
    if shutil.which("nvcc") is None:
        print(
            "WARNING: nvcc not found. CUDA C++ extensions "
            f"({label or ', '.join(extensions)}) were skipped. "
            "On HPC environments (e.g. Euler), load CUDA module first: "
            "module load stack/2024-06 gcc/12.2.0 cuda/12.4.1"
        )
        return

    os.environ["FORCE_CUDA"] = "1"
    if "CUDA_HOME" not in os.environ:
        nvcc_path = shutil.which("nvcc")
        if nvcc_path:
            os.environ["CUDA_HOME"] = str(Path(nvcc_path).resolve().parents[1])
    if "CUB_HOME" not in os.environ and "CUDA_HOME" in os.environ:
        os.environ["CUB_HOME"] = os.environ["CUDA_HOME"]
    if "MAX_JOBS" not in os.environ:
        os.environ["MAX_JOBS"] = "8"

    environment = PROJECT_ROOT / "third_party/.venvs" / method
    print(f"Installing CUDA extensions ({label or ', '.join(extensions)})...")
    for extension in extensions:
        subprocess.run(
            [
                "uv",
                "pip",
                "install",
                "--python",
                str(environment / "bin/python"),
                "--no-build-isolation",
                extension,
            ],
            check=True,
        )
