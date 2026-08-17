#!/usr/bin/env bash
#SBATCH --job-name=material_anything_2d_full
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=rtx_4090:1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=8G
#SBATCH --time=24:00:00
#SBATCH --output=logs/sbatch/material_anything_2d_full-%j.out
#SBATCH --error=logs/sbatch/material_anything_2d_full-%j.err

set -euo pipefail

export HF_HOME="/cluster/scratch/xiwang1/.cache/huggingface"
export TRANSFORMERS_CACHE="${HF_HOME}"

module load stack/2024-06 gcc/12.2.0 cuda/12.8.0 2>/dev/null || true

readonly PROJECT_ROOT="/cluster/scratch/xiwang1/hiwi/PBREstimationEval"
readonly PYTHON="${PROJECT_ROOT}/third_party/.venvs/material_anything/bin/python"
readonly PREDICTIONS_DIR="${PROJECT_ROOT}/outputs/pbr_2d/material_anything/predictions"

cd "${PROJECT_ROOT}"
mkdir -p logs/sbatch

if [[ ! -x "${PYTHON}" ]]; then
    echo "Material Anything environment not found: ${PYTHON}" >&2
    echo "Create it with: uv run python scripts/setup/material_anything_deps.py" >&2
    exit 2
fi

echo "========================================================"
echo " Stage 1: Material Anything 2D Inference"
echo "========================================================"
"${PYTHON}" -u src/infer_pbr_2d.py \
    method_2d=material_anything \
    data=all_2d \
    "$@"

echo "========================================================"
echo " Stage 2: Direct Evaluation (2D Metrics)"
echo "========================================================"
uv run python -u src/eval_pbr_2d_direct.py \
    predictions_dir="${PREDICTIONS_DIR}"

echo "========================================================"
echo " Stage 3: Indirect Evaluation (Blender Re-rendering)"
echo "========================================================"
module load blender/3.4.1 2>/dev/null || true

uv run python -u src/eval_pbr_2d_indirect.py \
    predictions_dir="${PREDICTIONS_DIR}" \
    save_rerenders=true

echo "========================================================"
echo " All stages completed successfully."
echo "========================================================"
