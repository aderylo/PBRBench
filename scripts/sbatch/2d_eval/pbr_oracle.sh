#!/usr/bin/env bash
#SBATCH --job-name=pbr_oracle_2d_full
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=rtx_4090:1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=8G
#SBATCH --time=24:00:00
#SBATCH --output=logs/sbatch/pbr_oracle_2d_full-%j.out
#SBATCH --error=logs/sbatch/pbr_oracle_2d_full-%j.err

set -euo pipefail

readonly PROJECT_ROOT="/cluster/scratch/xiwang1/hiwi/PBREstimationEval"
readonly PREDICTIONS_DIR="${PROJECT_ROOT}/outputs/pbr_2d/pbr_oracle/predictions"

cd "${PROJECT_ROOT}"
mkdir -p logs/sbatch

module load blender/3.4.1 2>/dev/null || true

if ! command -v blender >/dev/null 2>&1; then
    echo "Blender is unavailable after 'module load blender'." >&2
    exit 2
fi

echo "========================================================"
echo " Stage 1: PBR Oracle 2D Inference (Ground Truth)"
echo "========================================================"
uv run python -u src/infer_pbr_2d.py \
    method_2d=pbr_oracle \
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
uv run python -u src/eval_pbr_2d_indirect.py \
    predictions_dir="${PREDICTIONS_DIR}" \
    save_rerenders=true

echo "========================================================"
echo " All stages completed successfully."
echo "========================================================"
