"""Create the uv environment for MaterialAnything."""

import os
import shutil
import subprocess

from _setup import PROJECT_ROOT, setup_method_cli

PYTORCH3D_REQUIREMENT = (
    "pytorch3d @ git+https://github.com/facebookresearch/pytorch3d.git"
)


if __name__ == "__main__":
    if shutil.which("nvcc") is None:
        raise RuntimeError(
            "MaterialAnything requires a CUDA 12.8 compiler to build PyTorch3D. "
            "On Euler, run: module load stack/2024-06 gcc/12.2.0 cuda/12.8.0"
        )
    os.environ["FORCE_CUDA"] = "1"
    setup_method_cli(
        "material_anything",
        "third_party/MaterialAnything",
        bootstrap_requirements=(
            "torch==2.8.0",
            "torchvision==0.23.0",
            "setuptools<81",
            "wheel",
            "ninja",
        ),
        exclude_requirements="scripts/setup/material_anything-excludes.txt",
        extra_requirements=(
            "torch==2.8.0",
            "torchvision==0.23.0",
            "kaolin==0.18.0",
            "xformers==0.0.32.post2",
            "xatlas==0.0.11",
            "cupy-cuda12x==14.1.1",
            "scikit-image==0.26.0",
            "huggingface-hub==0.25.2",
            "transformers==4.41.2",
            "setuptools<81",
        ),
        torch_backend="cu128",
        find_links=(
            "https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-2.8.0_cu128.html",
        ),
    )
    # uv's wheel cache does not distinguish CPU and CUDA extension builds.
    # Force a fresh build so an older CPU-only PyTorch3D wheel cannot leak in.
    subprocess.run(
        [
            "uv",
            "pip",
            "install",
            "--python",
            str(PROJECT_ROOT / "third_party/.venvs/material_anything/bin/python"),
            "--reinstall",
            "--no-cache",
            "--no-build-isolation",
            PYTORCH3D_REQUIREMENT,
        ],
        check=True,
    )
