# DiffusionRenderer 2D adapter

The benchmark wraps the official [DiffusionRenderer](https://github.com/nv-tlabs/diffusion-renderer) single-image and video inverse rendering API in the `third_party/diffusion-renderer` submodule. It consumes screen-space observations and produces screen-space base color (albedo), roughness, metallic, and normal maps.

Upstream repository: <https://github.com/nv-tlabs/diffusion-renderer>

## Environment and Checkpoint Setup

1. Create the isolated virtual environment and download the model weights into
   `.weights/diffusion_renderer`:
```bash
uv run python scripts/setup/diffusion_renderer_deps.py
uv run python scripts/setup/diffusion_renderer_weights.py
```

## Benchmark Command

Run 2D inference using the method-specific environment:

```bash
third_party/.venvs/diffusion_renderer/bin/python src/infer_pbr_2d.py \
  method_2d=diffusion_renderer
```

The default dataset is the prepared TexVerse dataset at
`data/2d_eval/texverse`.

To restrict a development or debugging run to a few samples:

```bash
third_party/.venvs/diffusion_renderer/bin/python src/infer_pbr_2d.py \
  method_2d=diffusion_renderer data.max_samples=2
```
