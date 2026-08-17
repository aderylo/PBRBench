# SuperMat 2D adapter

The benchmark wraps the official SuperMat single-image Python API pinned
at commit `288339684d2d1d12283ee7c373fcce28982aba7c` in the
`third_party/SuperMat` submodule. It consumes the canonical 512 × 512
RGBA observation and produces screen-space albedo, perceptual roughness, and
metallic maps.

Upstream repository: <https://github.com/hyj542682306/SuperMat>. The pinned
repository does not contain a license file; do not assume redistribution or
commercial-use rights without clarification from the authors.

SuperMat's released multi-view model also produces screen-space maps. Its UV
refiner requires an existing projected UV map, while the released repository
does not provide the required view-to-UV fusion stage. SuperMat is therefore
registered only as a 2D method.

## Environment and checkpoint

```bash
uv run python scripts/setup/supermat_deps.py
uv run python scripts/setup/supermat_weights.py
```

The setup command creates `third_party/.venvs/supermat` and installs both the
small benchmark runtime and SuperMat's requirements. uv reuses its global
package cache across method environments. The weights command downloads
`supermat.pth` from the `oyiya/SuperMat` Hugging Face repository into
`.weights/supermat`.

The adapter uses `sd2-community/stable-diffusion-2-1` by default because the
original upstream base-model location is unavailable. Override
`method_2d.base_model` with a local directory for offline execution.

## Benchmark command

```bash
third_party/.venvs/supermat/bin/python src/infer_pbr_2d.py method_2d=supermat
```

The upstream albedo PNG is preserved as sRGB-encoded base color. Upstream
roughness and metallic outputs are converted to scalar PNGs without changing
their perceptual-roughness or metallic conventions.

The estimator loads the pinned upstream module in the current process and calls
`build_pipeline()` and `run_one_image()` directly. SuperMat names its own package
`src`, so the adapter extends this project's `src` namespace with SuperMat's
package path before importing it. This method-specific compatibility detail is
contained entirely in the adapter.
