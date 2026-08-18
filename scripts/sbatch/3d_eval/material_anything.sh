#!/usr/bin/env bash
#SBATCH --job-name=material_anything_3d_full
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=rtx_4090:1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=8G
#SBATCH --time=24:00:00
#SBATCH --output=logs/sbatch/material_anything_3d_full-%j.out
#SBATCH --error=logs/sbatch/material_anything_3d_full-%j.err

set -euo pipefail

readonly PROJECT_ROOT="/cluster/scratch/xiwang1/hiwi/PBREstimationEval"
export HF_HOME="${PROJECT_ROOT}/.weights/huggingface"
export TRANSFORMERS_CACHE="${HF_HOME}"
export TORCH_HOME="${PROJECT_ROOT}/.weights/torch"
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1


# Load CUDA stack matching PyTorch cu128 build
module load stack/2024-06 gcc/12.2.0 cuda/12.8.0 2>/dev/null || true

readonly PYTHON="${PROJECT_ROOT}/third_party/.venvs/material_anything/bin/python"
readonly PREDICTIONS_DIR="${PROJECT_ROOT}/outputs/pbr_3d/material_anything/predictions"

cd "${PROJECT_ROOT}"
mkdir -p logs/sbatch

if [[ ! -x "${PYTHON}" ]]; then
    echo "Material Anything environment not found: ${PYTHON}" >&2
    echo "Create it with: uv run python scripts/setup/material_anything_deps.py" >&2
    exit 2
fi

echo "========================================================"
echo " Stage 1: Material Anything 3D Inference"
echo "========================================================"
"${PYTHON}" -u src/infer_pbr_3d.py \
    method_3d=material_anything \
    data=all_3d \
    "$@"

echo "========================================================"
echo " Stage 2: Direct Evaluation (3D Metrics)"
echo "========================================================"
uv run python -u src/eval_pbr_3d_direct.py \
    predictions_dir="${PREDICTIONS_DIR}"

echo "========================================================"
echo " Stage 3: Indirect Evaluation (Blender Re-rendering)"
echo "========================================================"
module load blender/3.4.1 2>/dev/null || true

uv run python -u src/eval_pbr_3d_indirect.py \
    predictions_dir="${PREDICTIONS_DIR}" \
    save_rerenders=true

echo "========================================================"
echo " All stages completed successfully."
echo "========================================================"
