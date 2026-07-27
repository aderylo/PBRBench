"""Shared utilities for preparing method-specific environments."""

from __future__ import annotations

import argparse
import subprocess
import tempfile
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _normalize_legacy_editables(
    requirements: Path,
) -> tuple[Path, tuple[str, ...], Path | None]:
    """Convert legacy editable Git lines into uv-compatible direct references."""
    kept_lines: list[str] = []
    direct_requirements: list[str] = []

    for line in requirements.read_text().splitlines(keepends=True):
        stripped = line.strip()
        if stripped.startswith("-e git+") and "#egg=" in stripped:
            package = stripped.split("#egg=", 1)[1].split("&", 1)[0]
            git_url = stripped[3:].split("#", 1)[0]
            direct_requirements.append(f"{package} @ {git_url}")
        else:
            kept_lines.append(line)

    if not direct_requirements:
        return requirements, (), None

    temporary = tempfile.NamedTemporaryFile(
        mode="w",
        prefix="pbr-eval-requirements-",
        suffix=".txt",
        delete=False,
    )
    try:
        temporary.write("".join(kept_lines))
    finally:
        temporary.close()

    temporary_path = Path(temporary.name)
    return temporary_path, tuple(direct_requirements), temporary_path


def setup_method(
    method: str,
    repository: str,
    *,
    python: str = "3.11",
    bootstrap_requirements: tuple[str, ...] = (),
    exclude_requirements: str | None = None,
    extra_requirements: tuple[str, ...] = (),
    index_strategy: str | None = None,
    torch_backend: str | None = None,
    find_links: tuple[str, ...] = (),
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
                *(
                    ("--torch-backend", torch_backend)
                    if torch_backend is not None
                    else ()
                ),
            ],
            check=True,
        )

    install_requirements, converted_editables, temporary_requirements = (
        _normalize_legacy_editables(requirements)
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
    bootstrap_requirements: tuple[str, ...] = (),
    exclude_requirements: str | None = None,
    extra_requirements: tuple[str, ...] = (),
    torch_backend: str | None = None,
    find_links: tuple[str, ...] = (),
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
        torch_backend=torch_backend,
        find_links=find_links,
        no_build_isolation_packages=no_build_isolation_packages,
    )
