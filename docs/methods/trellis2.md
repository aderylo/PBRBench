# TRELLIS 2 3D PBR Material Texturing Adapter

The benchmark wraps Microsoft's [TRELLIS.2](https://github.com/microsoft/TRELLIS.2) 3D foundation model in the `third_party/TRELLIS.2` submodule. Specifically, the adapter uses `Trellis2TexturingPipeline` to recover PBR materials (Albedo, Perceptual Roughness, and Metallic) for a target 3D mesh given a single reference image observation.

Upstream repository: <https://github.com/microsoft/TRELLIS.2>

## Method Overview

Given a 3D geometry mesh and a single reference image:
1. **Conditioning & Voxelization**: TRELLIS 2 extracts visual appearance features (DINOv2) and voxelizes geometry into a 3D sparse shape latent via `o_voxel`.
2. **Flow Matching Sampling**: A 3D Flow Matching model generates 3D texture latents (`tex_slat`) conditioned on shape and image features.
3. **PBR Decoding & Texture Rasterization**: The texture latent is decoded into 3D PBR voxels (Base Color, Metallic, Roughness, Opacity), rasterized into UV space using `nvdiffrast` and `CuMesh`, and saved as glTF standard PBR material textures.
4. **Benchmark Map Extraction**: The adapter extracts Albedo (base color RGB), Roughness (metallicRoughness Green channel), and Metallic (metallicRoughness Blue channel) into standardized PNG maps alongside the textured `.glb` mesh output.

## Environment Setup

Create the isolated virtual environment and build required CUDA C++ extensions (`nvdiffrast`, `CuMesh`, `FlexGEMM`, `o-voxel`):

```bash
uv run python scripts/setup/trellis2_deps.py
```

*Note*: On HPC clusters (e.g. Euler), load CUDA compiler tools before running setup:
```bash
module load gcc/12.2.0 cuda/12.4.0
uv run python scripts/setup/trellis2_deps.py
```

## Checkpoint Downloads

Model weights (`microsoft/TRELLIS.2-4B`) are downloaded into
`.weights/trellis2`. To pre-download them:

```bash
uv run python scripts/setup/trellis2_weights.py
```

## Benchmark Execution Command

Run 3D inference using the method-specific environment:

```bash
third_party/.venvs/trellis2/bin/python src/infer_pbr_3d.py method_3d=trellis2
```

To run inference on a small subset of samples (e.g. 2 samples for testing):

```bash
third_party/.venvs/trellis2/bin/python src/infer_pbr_3d.py method_3d=trellis2 data.max_samples=2
```
