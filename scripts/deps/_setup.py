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
    extra_requirements: tuple[str, ...] = (),
) -> None:
    """Create a uv environment containing the benchmark and method dependencies."""
    repository_root = PROJECT_ROOT / repository
    requirements = repository_root / "requirements.txt"
    if not requirements.is_file():
        raise FileNotFoundError(requirements)

    environment = PROJECT_ROOT / "third_party/.venvs" / method
    subprocess.run(
        ["uv", "venv", "--python", python, str(environment)], check=True
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
        ],
        check=True,
    )
    print(f"Ready: {environment / 'bin/python'}")


def setup_method_cli(
    method: str,
    repository: str,
    *,
    extra_requirements: tuple[str, ...] = (),
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
        extra_requirements=extra_requirements,
    )
