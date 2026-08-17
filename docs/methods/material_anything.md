# Material Anything 3D adapter

The benchmark wraps the official Material Anything texture generation pipeline
in the `third_party/MaterialAnything` submodule. The adapter reuses the
upstream material estimator, progressive PBR-map conditioning, UV refiner, and
Voronoi hole filling, but skips the ControlNet/RePaint stages: every evaluated
object already provides a baked RGB texture, so the benchmark texture replaces
the upstream generated appearance texture.

Upstream repository: <https://github.com/NIRVANALAN/MaterialAnything>.

## Environment setup

Create the isolated virtual environment (PyTorch3D, kaolin, xformers, xatlas,
cupy) and build the PyTorch3D CUDA extension:

```bash
uv run python scripts/setup/material_anything_deps.py
```

*Note*: On HPC clusters (e.g. Euler), load CUDA 12.8 compiler tools before
running setup:

```bash
module load stack/2024-06 gcc/12.2.0 cuda/12.8.0
uv run python scripts/setup/material_anything_deps.py
```

## Checkpoint downloads

Download the material estimator and UV refiner weights into the submodule:

```bash
uv run python scripts/setup/material_anything_weights.py
```

By default the adapter expects them under
`.weights/material_anything/material_estimator` and
`.../material_refiner`; override `image2materials_model` / `uvrefine_model` in
`configs/method_3d/material_anything.yaml` if you store them elsewhere.
Relative model paths are resolved against the project root first and the
submodule second.

## Benchmark command

```bash
third_party/.venvs/material_anything/bin/python src/infer_pbr_3d.py method_3d=material_anything
```

## Method overview

The estimator implements the common 3D interface: `predict_over_dataset(samples,
output_dir)` takes the collection of prepared 3D samples and streams one
`Prediction3D` per sample (consumed one at a time by the driver, so runs stay
memory-bounded), organized into two dataset-wide passes:

1. **Pass A — Render, geometry & estimation**: the estimator model is loaded
   into VRAM once. For every sample the adapter normalizes the mesh with UVs,
   attaches the benchmark baked RGB texture, renders multi-view images and
   camera-space normals from the fixed `objaverse` viewpoint preset, builds the
   upstream similarity cache, rasterizes the canonical-coordinate map (CCM),
   runs the progressive estimator per view (conditioned on the rendered image,
   normal, and previously backprojected materials), backprojects each view into
   UV space, and bakes the multi-view predictions into coarse UV maps. The
   estimator is unloaded afterwards.
2. **Pass B — UV refinement & hole filling**: the UV refiner is loaded into
   VRAM once, then each sample is processed one by one: the CCM-conditioned
   refiner improves the coarse maps and Voronoi propagation fills unmapped UV
   texels. The final channels are written directly into the standard output
   layout (see below), intermediates are cleaned up, and the refiner is
   unloaded.

The estimator and refiner are never co-resident in VRAM, and each model is
loaded only once per run. Both run 50 diffusion steps with guidance scale 1.0
and a fixed seed (`configs/method_3d/material_anything.yaml`). The optional
text `prompt` is ignored because the baked-texture path uses an empty prompt.

## Output layout

Standard prediction channels under `<output_dir>/<sample_id>/`:

| File | Content |
| --- | --- |
| `albedo.png` | base color (RGB) |
| `roughness.png` | perceptual roughness (green channel of the upstream R/M map) |
| `metallic.png` | metallic (blue channel of the upstream R/M map) |

The method writes these directly into the standard layout during pass B,
bakes them into `mesh.glb`, and returns the paths via `Prediction3D`. The
driver then calls `BaseMaterialEstimator3D.align_to_original_uv`, which with
the identity UV correspondence used by this adapter copies the returned
`mesh.glb` to `canonical_asset.glb`, so no prediction is stored twice.

Intermediate stage results are written under
`<sample_id>/intermediate/` as method-internal scratch:

| Directory | Contents |
| --- | --- |
| `stage1/` | `view_NN_image.png`, `view_NN_normal.png`, `uv_ccm.png`, `uv_mask.png` |
| `stage2/` | `view_NN_albedo.png`, `view_NN_rm.png`, `view_NN_bump.png`, `view_NN_mask.png`, `coarse_albedo_uv.png`, `coarse_rm_uv.png`, `coarse_bump_uv.png` |
| `stage3/` | `refined_albedo_uv.png`, `refined_rm_uv.png`, `refined_bump_uv.png`, `final_albedo_uv.png`, `final_rm_uv.png`, `final_bump_uv.png` |

`intermediate/` is deleted after each sample finishes unless
`cleanup_intermediates: false` (equivalently `save_intermediates: true`) is
set. When kept, the key maps (`ccm`, `coarse_*`, `refined_*`) are also
hardlinked into the sample directory as prediction artifacts.

The resolved run configuration is saved once per run at
`<output_dir>/config.yaml`.
