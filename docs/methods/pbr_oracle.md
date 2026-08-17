# PBROracle diffuse-albedo estimator

This benchmark method renders an expanded controlled sequence with the
authored PBR material and a matched neutral-Lambertian shading proxy for every
view/environment pair. The default sequence contains twelve states: four HDR
environments at three well-separated rotations and reduced exposure. It
estimates diffuse albedo by dividing each relighting by its proxy in linear RGB
and taking a per-channel soft-weighted median in log space.

Proxy energy provides an inverse-variance-style reliability weight. Values
near LDR clipping are smoothly downweighted but never discarded, preventing
the former saturation-mask holes. `confidence.png` records the accumulated
soft photometric support.

## Run

Load Blender in the GPU allocation, then launch the common inference entry
point from the benchmark environment:

```bash
module load blender
uv run python src/infer_pbr_2d.py method_2d=pbr_oracle
```

For a small development run:

```bash
uv run python src/infer_pbr_2d.py \
  method_2d=pbr_oracle \
  data.max_samples=2
```

The method groups pending samples by object/view, so all configured light
observations contribute to the same albedo even if only one sample from that
view is pending. Relightings, proxies, normalized previews, coverage, and the
linear estimate are cached below
`outputs/pbr_2d/pbr_oracle/oracle_cache`.

`albedo.png` is the meaningful output. `roughness.png` is a constant 0.5 and
`metallic.png` is a constant 0.0 within the supported object mask so the method
can use the existing three-channel benchmark contract. The albedo estimate is
diffuse albedo and is not PBR base color on metallic surfaces.
