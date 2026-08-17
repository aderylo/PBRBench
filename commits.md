# Commit train for the current working tree

Order matches your request: **existing 2D adaptation first → new 2D methods (one by one) → 3D infra → 3D methods (one by one)**. Every commit is meant to leave the repo in a working state (staged set is self-consistent, imports resolve, committed eval scripts keep working).

> ⚠️ **Read the "Bugs that MUST be fixed first" section at the bottom before approving.** A few files will *not* run as committed unless patched (missing imports, missing `dataset.name`). The plan assumes those fixes are applied as part of the affected commit.

---

## Phase 1 — Existing 2D adaptation

### C1 ✅ `refactor(2d): centralize method setup under scripts/setup, add weight downloads` — **committed** (`43c380f`)
- `.gitignore` (add `.weights/`)
- `README.md` (setup script renames; also mentions IDArb, harmless since it lands in this batch)
- `configs/method_2d/diffusion_renderer.yaml`, `neural_lightrig.yaml`, `supermat.yaml` (checkpoints → `.weights/...`)
- `src/methods_2d/diffusion_renderer.py`, `src/methods_2d/neural_lightrig.py` (checkpoint dir resolution)
- `scripts/setup/_setup.py` (moved from `scripts/deps/` + `install_cuda_extensions` etc.)
- `scripts/setup/diffusion_renderer-excludes.txt`, `diffusion_renderer_deps.py`, `diffusion_renderer_weights.py`
- `scripts/setup/neural_lightrig_deps.py`, `neural_lightrig_weights.py`
- `scripts/setup/supermat_deps.py`, `supermat_weights.py`

```bash
git add .gitignore README.md
git add configs/method_2d/diffusion_renderer.yaml configs/method_2d/neural_lightrig.yaml configs/method_2d/supermat.yaml
git add src/methods_2d/diffusion_renderer.py src/methods_2d/neural_lightrig.py
git add scripts/setup/_setup.py scripts/setup/diffusion_renderer-excludes.txt \
        scripts/setup/diffusion_renderer_deps.py scripts/setup/diffusion_renderer_weights.py \
        scripts/setup/neural_lightrig_deps.py scripts/setup/neural_lightrig_weights.py \
        scripts/setup/supermat_deps.py scripts/setup/supermat_weights.py
```

### C2 `feat(2d): per-method sbatch scripts, method docs, single-view infer scope`
- `src/methods_2d/base.py`
- `scripts/sbatch/2d_infer/{diffusion_renderer,neural_lightrig,pbr_oracle,supermat}.sh` (new)
- `scripts/sbatch/eval_pbr_2d_direct.sh`, `scripts/sbatch/eval_pbr_2d_indirect.sh`, `scripts/sbatch/render_eval_data_2d.sh` (new)
- delete `scripts/sbatch/indirect_eval_2d.sh`, `scripts/sbatch/pbr_oracle.sh`, `scripts/sbatch/supermat.sh`
- `docs/methods/{diffusion_renderer,neural_lightrig,pbr_oracle,supermat}.md`
- `configs/infer_pbr_2d.yaml` (`view_ids: [view_00]`)

```bash
git add src/methods_2d/base.py

git add scripts/sbatch/2d_infer/diffusion_renderer.sh scripts/sbatch/2d_infer/neural_lightrig.sh \
        scripts/sbatch/2d_infer/pbr_oracle.sh scripts/sbatch/2d_infer/supermat.sh
git add scripts/sbatch/eval_pbr_2d_direct.sh scripts/sbatch/eval_pbr_2d_indirect.sh \
        scripts/sbatch/render_eval_data_2d.sh
git rm scripts/sbatch/indirect_eval_2d.sh scripts/sbatch/pbr_oracle.sh scripts/sbatch/supermat.sh
git add docs/methods/diffusion_renderer.md docs/methods/neural_lightrig.md \
        docs/methods/pbr_oracle.md docs/methods/supermat.md
git add configs/infer_pbr_2d.yaml
```

### C3 `refactor(2d): simplify multi-source dataset config (sources -> roots)`
- `src/data/pbr_estimation_dataset_2d.py` + `configs/data/all_2d.yaml`
- **Must also fix**: restore a `name` attribute on `MultiSourcePBREstimationDataset2D` (the committed `eval_pbr_2d_direct.py` / `eval_pbr_2d_indirect.py` call `dataset.name` — the refactor removes it and would crash them).

```bash
git add src/data/pbr_estimation_dataset_2d.py configs/data/all_2d.yaml
```

---

## Phase 2 — New 2D methods (one commit each)

### C4 `feat(2d): add IDArb single-view intrinsic-decomposition adapter`
- `src/methods_2d/idarb.py`, `configs/method_2d/idarb.yaml`, `scripts/sbatch/2d_infer/idarb.sh`, `docs/methods/idarb.md`
- `scripts/setup/idarb_deps.py`, `idarb_weights.py`, `idarb-excludes.txt`
- submodule gitlink `third_party/IDArb` + `.gitmodules` entry (IDArb hunk only)

```bash
git add src/methods_2d/idarb.py configs/method_2d/idarb.yaml scripts/sbatch/2d_infer/idarb.sh docs/methods/idarb.md
git add scripts/setup/idarb_deps.py scripts/setup/idarb_weights.py scripts/setup/idarb-excludes.txt
git add third_party/IDArb
git add -p .gitmodules    # stage ONLY the [submodule "third_party/IDArb"] hunk
```

### C5 `feat(2d): add IntrinsicAnything diffusion-prior adapter`
- `src/methods_2d/intrinsic_anything.py`, `configs/method_2d/intrinsic_anything.yaml`, `docs/methods/intrinsic_anything.md`
- `scripts/setup/intrinsic_anything_deps.py`, `intrinsic_anything_weights.py`, `intrinsic_anything-excludes.txt`

```bash
git add src/methods_2d/intrinsic_anything.py configs/method_2d/intrinsic_anything.yaml \
        docs/methods/intrinsic_anything.md
git add scripts/setup/intrinsic_anything_deps.py scripts/setup/intrinsic_anything_weights.py \
        scripts/setup/intrinsic_anything-excludes.txt
```

### C6 `feat(2d): add Material Anything single-view adapter + shared env setup`
- `src/methods_2d/material_anything.py`, `configs/method_2d/material_anything.yaml`, `scripts/sbatch/2d_infer/material_anything.sh`
- `scripts/setup/material_anything_deps.py`, `material_anything_weights.py`, `material_anything-excludes.txt` (shared with the 3D adapter — lands here, per your note)

```bash
git add src/methods_2d/material_anything.py configs/method_2d/material_anything.yaml \
        scripts/sbatch/2d_infer/material_anything.sh
git add scripts/setup/material_anything_deps.py scripts/setup/material_anything_weights.py \
        scripts/setup/material_anything-excludes.txt
```

---

## Phase 3 — 3D

### C7 `feat(3d): 3D dataset, estimator base, shared utils, infer/eval entrypoints`
- `src/data/pbr_estimation_dataset_3d.py` (**add `name` — both committed `eval_pbr_3d_direct.py` and new `eval_pbr_3d_indirect.py` call `dataset.name`**)
- `src/methods_3d/__init__.py`, `src/methods_3d/base.py`
- `src/utils/glb.py`, `rendering.py`, `split.py`, `_relight_pbr_3d_blender.py`, `__init__.py` (re-exports), `metrics.py` (`masked_mae_rmse_psnr` — required by committed `eval_pbr_3d_direct.py`)
- `src/infer_pbr_3d.py`, `src/eval_pbr_3d_indirect.py` (**fix missing `numpy` / `Any` imports**)
- `configs/data/{all_3d,dtc_3d,objaverse_3d,polyhaven_3d,texverse_3d}.yaml`
- `configs/infer_pbr_3d.yaml`, `configs/eval_pbr_3d_direct.yaml`, `configs/eval_pbr_3d_indirect.yaml` (new)
- `scripts/sbatch/render_eval_data_3d.sh`
- `pyproject.toml` (`trimesh`) + `uv.lock`

```bash
git add src/data/pbr_estimation_dataset_3d.py
git add src/methods_3d/__init__.py src/methods_3d/base.py
git add src/utils/glb.py src/utils/rendering.py src/utils/split.py \
        src/utils/_relight_pbr_3d_blender.py src/utils/__init__.py src/utils/metrics.py
git add src/infer_pbr_3d.py src/eval_pbr_3d_indirect.py
git add configs/data/all_3d.yaml configs/data/dtc_3d.yaml configs/data/objaverse_3d.yaml \
        configs/data/polyhaven_3d.yaml configs/data/texverse_3d.yaml
git add configs/infer_pbr_3d.yaml configs/eval_pbr_3d_direct.yaml configs/eval_pbr_3d_indirect.yaml
git add scripts/sbatch/render_eval_data_3d.sh
git add pyproject.toml uv.lock
```

### C8 `feat(3d): add Material Anything in-process bake-texture adapter`
- `src/methods_3d/material_anything.py` (**fix missing `create_textured_glb` import — NameError at runtime otherwise**)
- `configs/method_3d/material_anything.yaml`, `scripts/sbatch/3d_infer/material_anything.sh`
- `docs/methods/material_anything.md` (fix the "`BaseMaterialEstimator3D.save_prediction`" claim — that method does not exist)

```bash
git add src/methods_3d/material_anything.py configs/method_3d/material_anything.yaml \
        scripts/sbatch/3d_infer/material_anything.sh docs/methods/material_anything.md
```

### C9 `feat(3d): add TRELLIS 2 texturing adapter`
- `src/methods_3d/trellis2.py`, `configs/method_3d/trellis2.yaml`, `scripts/sbatch/3d_infer/trellis2.sh`, `docs/methods/trellis2.md`
- `scripts/setup/trellis2_deps.py`, `trellis2_weights.py`, `trellis2-requirements.txt`

```bash
git add src/methods_3d/trellis2.py configs/method_3d/trellis2.yaml scripts/sbatch/3d_infer/trellis2.sh \
        docs/methods/trellis2.md
git add scripts/setup/trellis2_deps.py scripts/setup/trellis2_weights.py scripts/setup/trellis2-requirements.txt
```

### C10 `feat(3d): add Hunyuan3D-2.1 (Paint) texturing adapter`
- `src/methods_3d/hunyuan3d.py`, `configs/method_3d/hunyuan3d.yaml`, `scripts/sbatch/3d_infer/hunyuan3d.sh`
- `scripts/setup/hunyuan3d_deps.py`, `hunyuan3d_weights.py`, `hunyuan3d-requirements.txt`
- submodule gitlink `third_party/Hunyuan3D-2.1` + `.gitmodules` entry (Hunyuan hunk only)

```bash
git add src/methods_3d/hunyuan3d.py configs/method_3d/hunyuan3d.yaml scripts/sbatch/3d_infer/hunyuan3d.sh
git add scripts/setup/hunyuan3d_deps.py scripts/setup/hunyuan3d_weights.py scripts/setup/hunyuan3d-requirements.txt
git add third_party/Hunyuan3D-2.1
git add -p .gitmodules    # stage ONLY the [submodule "third_party/Hunyuan3D-2.1"] hunk
```

---

## Bugs that MUST be fixed before/while staging (would otherwise break the committed repo) — ✅ all applied in the working tree

| # | Where | Problem | Fix |
|---|-------|---------|-----|
| 1 ✅ | `src/methods_3d/material_anything.py:764` | `create_textured_glb` used but never imported → NameError | add `from src.utils.glb import create_textured_glb` |
| 2 ✅ | `src/eval_pbr_3d_indirect.py:76` | `Any` in annotation, not imported | add `from typing import Any` |
| 3 ✅ | `src/eval_pbr_3d_indirect.py:269` | `np` used, numpy not imported | add `import numpy as np` |
| 4 ✅ | `src/data/pbr_estimation_dataset_3d.py` / `src/data/pbr_estimation_dataset_2d.py` | no `name` attribute; committed `eval_pbr_*` scripts call `dataset.name` → AttributeError | add a `name` property/attribute to both `MultiSource*Dataset` classes |
| 5 ✅ | `docs/methods/material_anything.md` | references non-existent `BaseMaterialEstimator3D.save_prediction` | reword to match actual flow (direct write + `align_to_original_uv`) |
| 6 ✅ | `src/utils/metrics.py:46` | triple blank line before `ssim` | drop one blank line |

## Minor lint (down from 41 ruff findings; the must-fix/critical ones are gone) — fix opportunistically, or leave as-is
- 15 × RUF100 unused `# noqa: E402` (E402 not enabled) — harmless
- 5 × I001 import sorting, 1 × F401 unused import (`PBREstimationDataset3D` in `infer_pbr_3d.py`; the `CHANNELS` one in `eval_pbr_3d_indirect.py` was removed with bug fix #2/#3)
- 5 × BLE001 blind `except Exception` in setup scripts — intentional
- a few RUF046 / FURB192 / RUF022 nits

## Do NOT stage (needs work / cannot be committed)

- **`docs/updates/update_13_08.tex`** — empty file (0 bytes). Either fill it or delete it.
- **Submodule dirty content** — the six submodules (TRELLIS.2, Neural-LightRig, MaterialAnything, SuperMat, diffusion-renderer, IntrinsicAnything) have uncommitted changes *inside* them (mostly deleted assets, but also real code edits: TRELLIS.2 `image_feature_extractor.py`/`BiRefNet.py`, Neural-LightRig `mld/pipeline.py`, MaterialAnything `download_models.sh`). The parent repo can only record the submodule commit SHA, so these edits are **not reproducible** from a fresh clone. If that matters: commit+push inside each submodule, then bump the gitlinks here. Otherwise they'll show `modified content` in `git status` forever.
- **Machine-specific hacks inside submodules** — the TRELLIS.2 patch hardcodes `/cluster/scratch/xiwang1/.cache/huggingface/...`; Neural-LightRig's `_resolve_hf_snapshot` uses `Path.home()`. These work locally but will break on another machine.
- **`third_party/.venvs/hunyuan3d`** — symlink into a *different* project (`pbr-estimation-post-training`). Env-only (gitignored) but fragile; the Hunyuan environment lives outside this repo.

## Notes / small gaps to be aware of

- **IntrinsicAnything has no sbatch script** (`scripts/sbatch/2d_infer/intrinsic_anything.sh` missing) while every other method has one — add for parity or accept the gap.
- `configs/data/all_2d.yaml` renamed `sources:` → `roots:`; only that config uses the multi-source 2D class, so nothing else needs updating.
- `scripts/setup/_setup.py` and the `trellis2_deps.py` / `hunyuan3d_deps.py` scripts reference `install_cuda_extensions` — present in the working tree version, make sure that version is the staged one.
- After C3, `eval_pbr_2d_*` keeps working only because of the `name` fix (bug #4) — don't skip it.