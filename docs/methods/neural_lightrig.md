# Neural LightRig 2D adapter

The benchmark wraps the official Neural LightRig Python API pinned at
commit `db472e191e7abd3115ced0d5c72d61b75a8412ed` in the
`third_party/Neural-LightRig` submodule. It consumes a background-removed
512 × 512 RGBA observation and predicts screen-space albedo, perceptual
roughness, and metallic maps. Its additional normal, reference-light, mask,
and combined outputs are retained under `native/` when `runtime.keep_native`
is enabled, but are not treated as canonical PBR channels.

Upstream repository: <https://github.com/ZexinHe/Neural-LightRig>. The released
code is licensed under Apache-2.0; checkpoint terms must still be checked at
their distribution source.

## Environment and checkpoint

```bash
uv run python scripts/setup/neural_lightrig_deps.py
```

This creates `third_party/.venvs/neural_lightrig` and installs both the small
benchmark runtime and Neural LightRig's requirements. uv reuses its global
package cache across method environments. The setup helper additionally installs
`torchvision`, which the released inference code imports but the upstream
`requirements.txt` omits.

The checkpoints are hosted in the gated Hugging Face repository
<https://huggingface.co/zxhezexin/neural-lightrig-mld-and-recon>. Request access
there, then authenticate once:

```bash
third_party/.venvs/neural_lightrig/bin/hf auth login
```

Then pre-download the checkpoints into `.weights/neural_lightrig`:

```bash
uv run python scripts/setup/neural_lightrig_weights.py
```

Inference downloads only `mld.pt` and `recon/**` into
`.weights/neural_lightrig`. The default config pins snapshot
`5619cfec5e623ded0701d0b05f26ad5bbf9f0401`. A complete local checkpoint is
used directly without contacting Hugging Face; its location can be overridden
with `method_2d.checkpoint_dir=/path/to/ckpt`.

The adapter uses the maintained community mirrors
`sd2-community/stable-diffusion-2-1` and
`sd2-community/stable-diffusion-2-1-unclip`, because the original Stability AI
repositories referenced by the released code are no longer available. The
model identifiers are configurable as `method_2d.base_model` and
`method_2d.unclip_model`.

## Benchmark command

```bash
third_party/.venvs/neural_lightrig/bin/python src/infer_pbr_2d.py \
  method_2d=neural_lightrig
```

Neural LightRig stores roughness in the green channel and metallic in the blue
channel of its native RM result. Its own conversion code separates those into
scalar maps before the benchmark adapter validates and saves them.

The published reconstruction config declares bfloat16 weights, while the
released `predict()` path constructs float32 inputs. The adapter restores the
reconstruction model to float32 after loading so its weights and inputs match.

The estimator prepends the pinned submodule to `sys.path`, loads both models
once in `setup()`, and calls the upstream preparation, inference, and conversion
functions directly. There is no worker process or dependency on the upstream
CLI. Run the common Hydra entrypoint with Neural LightRig's uv interpreter.
