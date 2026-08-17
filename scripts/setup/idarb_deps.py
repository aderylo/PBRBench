"""Create the uv environment for IDArb."""

from _setup import setup_method_cli

if __name__ == "__main__":
    setup_method_cli(
        "idarb",
        "third_party/IDArb",
        # Upstream pins PyTorch 2.2 / CUDA 11.8 tooling. Install the PyTorch
        # build required by the xformers release compiled for PyTorch 2.4.1.
        bootstrap_requirements=("torch==2.4.1", "setuptools", "wheel"),
        exclude_requirements="scripts/setup/idarb-excludes.txt",
        extra_requirements=(
            "torch==2.4.1",
            "torchvision==0.19.1",
            "xformers==0.0.28.post1",
            "transformers==4.44.2",
        ),
        # Drop the upstream cu118 extra-index so uv resolves everything from
        # its default index; all cu118-specific pins are excluded above.
        drop_index_lines=True,
        # Override upstream's xformers pin (compiled for PyTorch 2.2) with the
        # PyTorch 2.4.1 build supplied above.
        drop_requirements=("xformers",),
    )
