"""Create the uv environment for Hunyuan3D-2.1 (Hunyuan3D-Paint)."""

from __future__ import annotations

import shutil
import subprocess

from _setup import PROJECT_ROOT, install_cuda_extensions, setup_method_cli

REPOSITORY = "third_party/Hunyuan3D-2.1"

CUDA_EXTENSIONS = (
    "git+https://github.com/NVlabs/nvdiffrast.git@v0.4.0",
    "pytorch3d @ git+https://github.com/facebookresearch/pytorch3d.git",
)


if __name__ == "__main__":
    setup_method_cli(
        "hunyuan3d",
        REPOSITORY,
        # The upstream requirements file pins a stale PyTorch stack and
        # several demo-only dependencies, so the curated list shipped with
        # this script is installed instead.
        requirements="scripts/setup/hunyuan3d-requirements.txt",
        bootstrap_requirements=(
            "torch==2.6.0",
            "torchvision==0.21.0",
            "setuptools<81",
            "wheel",
        ),
        extra_requirements=("torch==2.6.0", "torchvision==0.21.0"),
        index_strategy="unsafe-best-match",
        torch_backend="cu124",
    )

    # hy3dpaint ships an uncompiled pybind11 extension for mesh inpainting.
    inpaint_cpp = (
        PROJECT_ROOT
        / REPOSITORY
        / "hy3dpaint/DifferentiableRenderer/mesh_inpaint_processor.cpp"
    )
    venv_python = PROJECT_ROOT / "third_party/.venvs/hunyuan3d/bin/python"
    if inpaint_cpp.is_file() and shutil.which("c++") is not None:
        print("Compiling mesh_inpaint_processor C++ extension...")
        try:
            pybind_includes = (
                subprocess.check_output(
                    [str(venv_python), "-m", "pybind11", "--includes"], text=True
                )
                .strip()
                .split()
            )
            ext_suffix = subprocess.check_output(
                [
                    str(venv_python),
                    "-c",
                    "import sysconfig; print(sysconfig.get_config_var('EXT_SUFFIX'))",
                ],
                text=True,
            ).strip()
            out_so = inpaint_cpp.parent / f"mesh_inpaint_processor{ext_suffix}"
            subprocess.run(
                [
                    "c++",
                    "-O3",
                    "-Wall",
                    "-shared",
                    "-std=c++11",
                    "-fPIC",
                    *pybind_includes,
                    str(inpaint_cpp),
                    "-o",
                    str(out_so),
                ],
                check=True,
            )
            print(f"Compiled inpaint extension to: {out_so}")
        except Exception as e:  # noqa: BLE001
            print(f"Warning: Failed to compile mesh_inpaint_processor: {e}")

    extensions = list(CUDA_EXTENSIONS)
    custom_rasterizer = PROJECT_ROOT / REPOSITORY / "hy3dpaint/custom_rasterizer"
    if custom_rasterizer.is_dir():
        extensions.append(str(custom_rasterizer))
    install_cuda_extensions(
        "hunyuan3d",
        tuple(extensions),
        label="nvdiffrast, PyTorch3D, custom_rasterizer",
    )
