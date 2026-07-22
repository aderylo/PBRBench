"""Create the uv environment for Neural LightRig."""

from _setup import setup_method_cli


if __name__ == "__main__":
    setup_method_cli(
        "neural_lightrig",
        "third_party/Neural-LightRig",
        # Imported by upstream code but omitted from its requirements file.
        extra_requirements=("torchvision",),
    )
