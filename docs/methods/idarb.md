# IDArb 2D adapter

The benchmark wraps the official IDArb single-image inference API pinned at
commit `df26b90e6ea47b803e1281fd9eed63cf00bb11a4` in the `third_party/IDArb`
submodule. It consumes the RGB observation and optional foreground mask and
produces screen-space albedo, normal, metallic, and perceptual roughness maps.

Upstream repository: <https://github.com/Lizb6626/IDArb>, released under the
MIT license (ICLR 2025).

The adapter implements only the released single-image workflow. IDArb's
multi-view mode consumes sets of registered views and is therefore out of
scope for the 2D benchmark task.

## Environment and checkpoint

```bash
uv run python scripts/setup/idarb_deps.py
uv run python scripts/setup/idarb_weights.py
```

The setup command creates `third_party/.venvs/idarb` and installs the
benchmark runtime together with IDArb's inference dependencies. The upstream
`requirements.txt` pins PyTorch 2.2 / CUDA 11.8 tooling; the setup script
installs PyTorch 2.4.1 with the matching xformers release
(`0.0.28.post1`, compiled for PyTorch 2.4.1). xformers is mandatory:
IDArb's joint-domain attention processor calls `xformers.ops` directly, and
the setup script therefore excludes upstream's stale `xformers==0.0.25` pin.
Training-only dependencies with incompatible compiled wheels (bitsandbytes,
pytorch-lightning, decord, nerfacc, and friends) are excluded as well.

The weights command downloads the diffusion components (`unet`, `vae`,
`text_encoder`, `tokenizer`, `feature_extractor`, `scheduler`, about 5.4 GB)
from the `lizb6626/IDArb` Hugging Face repository into
`.weights/idarb`.

## Benchmark command

```bash
third_party/.venvs/idarb/bin/python src/infer_pbr_2d.py method_2d=idarb
```

The estimator builds the upstream `IDArbDiffusionPipeline` from the local
checkpoints, keeps it resident, and processes the batch in chunks of
`method_2d.batch_size`.

## Input and output conventions

IDArb operates at a fixed 512-pixel canvas. The adapter replicates the
upstream `CustomDataset` preprocessing: the observation's longer side is
resized to 512 pixels (BICUBIC), zero-padded onto a 512 x 512 canvas, and
the foreground (from the sample mask, or the alpha channel when no mask is
registered) is composited over a white background. The camera pose fed to
the model is the fixed upstream single-image pose encoding.

The pipeline returns three decoded domains: albedo (RGB), normal (RGB), and
mro, whose channels 0 and 1 carry metallic and roughness. The adapter crops
the canvas padding back off and resizes each channel to the original
observation resolution. The upstream 512-pixel albedo and normal
predictions are retained as-is (up to resampling); metallic and roughness
are stored as scalar PNGs without changing their conventions.

The estimator imports the pinned upstream `idarbdiffusion` package directly
after placing the checkout on `sys.path`, and requires xformers at setup
time.
