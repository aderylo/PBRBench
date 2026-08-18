#!/usr/bin/env bash
#SBATCH --job-name=neural_lightrig_2d_full
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=rtx_4090:1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=8G
#SBATCH --time=24:00:00
#SBATCH --output=logs/sbatch/neural_lightrig_2d_full-%j.out
#SBATCH --error=logs/sbatch/neural_lightrig_2d_full-%j.err

set -euo pipefail

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

readonly PROJECT_ROOT="/cluster/scratch/xiwang1/hiwi/PBREstimationEval"
export HF_HOME="${PROJECT_ROOT}/.weights/huggingface"
export TORCH_HOME="${PROJECT_ROOT}/.weights/torch"
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
readonly PYTHON="${PROJECT_ROOT}/third_party/.venvs/neural_lightrig/bin/python"
readonly PREDICTIONS_DIR="${PROJECT_ROOT}/outputs/pbr_2d/neural_lightrig/predictions"

cd "${PROJECT_ROOT}"
mkdir -p logs/sbatch

if [[ ! -x "${PYTHON}" ]]; then
    echo "Neural LightRig environment not found: ${PYTHON}" >&2
    echo "Create it with: uv run python scripts/setup/neural_lightrig_deps.py" >&2
    exit 2
fi

echo "========================================================"
echo " Stage 1: Neural LightRig 2D Inference"
echo "========================================================"
"${PYTHON}" -u src/infer_pbr_2d.py \
    method_2d=neural_lightrig \
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
