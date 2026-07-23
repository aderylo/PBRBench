"""Create the uv environment for DiffusionRenderer."""

from _setup import setup_method_cli


if __name__ == "__main__":
    setup_method_cli(
        "diffusion_renderer",
        "third_party/diffusion-renderer",
        # Upstream tests PyTorch 2.1--2.4 and installs it before nvdiffrast.
        bootstrap_requirements=("torch==2.4.1", "setuptools", "wheel"),
        # The inverse-rendering pipeline does not use upstream's CUDA rasterizer.
        exclude_requirements="scripts/deps/diffusion_renderer-excludes.txt",
        extra_requirements=("torch==2.4.1", "torchvision==0.19.1"),
    )
