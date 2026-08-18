"""Create the uv environment for TRELLIS 2."""

from __future__ import annotations

from _setup import PROJECT_ROOT, install_cuda_extensions, setup_method_cli

REPOSITORY = "third_party/TRELLIS.2"

CUDA_EXTENSIONS = (
    "git+https://github.com/NVlabs/nvdiffrast.git@v0.4.0",
    "git+https://github.com/JeffreyXiang/CuMesh.git",
    "git+https://github.com/JeffreyXiang/FlexGEMM.git",
    "pytorch3d @ git+https://github.com/facebookresearch/pytorch3d.git",
)


if __name__ == "__main__":
    setup_method_cli(
        "trellis2",
        REPOSITORY,
        # The repository has no root requirements.txt; install the curated
        # list used by the benchmark instead.
        requirements="scripts/setup/trellis2-requirements.txt",
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

    extensions = list(CUDA_EXTENSIONS)
    o_voxel = PROJECT_ROOT / REPOSITORY / "o-voxel"
    if o_voxel.is_dir():
        extensions.append(str(o_voxel))
    install_cuda_extensions(
        "trellis2",
        tuple(extensions),
        label="nvdiffrast, CuMesh, FlexGEMM, PyTorch3D, o-voxel",
    )

    from trellis2_patch import apply_trellis2_patch

    apply_trellis2_patch()

