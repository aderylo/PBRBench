"""Shared utilities for preparing method-specific environments."""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def setup_method(
    method: str,
    repository: str,
    *,
    python: str = "3.11",
    bootstrap_requirements: tuple[str, ...] = (),
    exclude_requirements: str | None = None,
    extra_requirements: tuple[str, ...] = (),
    no_build_isolation_packages: tuple[str, ...] = (),
) -> None:
    """Create a uv environment containing the benchmark and method dependencies."""
    repository_root = PROJECT_ROOT / repository
    requirements = repository_root / "requirements.txt"
    if not requirements.is_file():
        raise FileNotFoundError(requirements)

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
            ],
            check=True,
        )
    subprocess.run(
        [
            "uv",
            "pip",
            "install",
            "--python",
            str(environment / "bin/python"),
            "-r",
            str(PROJECT_ROOT / "pyproject.toml"),
            "-r",
            str(requirements),
            *extra_requirements,
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
        ],
        check=True,
    )
    print(f"Ready: {environment / 'bin/python'}")


def setup_method_cli(
    method: str,
    repository: str,
    *,
    bootstrap_requirements: tuple[str, ...] = (),
    exclude_requirements: str | None = None,
    extra_requirements: tuple[str, ...] = (),
    no_build_isolation_packages: tuple[str, ...] = (),
) -> None:
    """Parse common command-line options and prepare a method environment."""
    parser = argparse.ArgumentParser(
        description=f"Create the uv environment for {method}."
    )
    parser.add_argument("--python", default="3.11")
    args = parser.parse_args()
    setup_method(
        method,
        repository,
        python=args.python,
        bootstrap_requirements=bootstrap_requirements,
        exclude_requirements=exclude_requirements,
        extra_requirements=extra_requirements,
        no_build_isolation_packages=no_build_isolation_packages,
    )
