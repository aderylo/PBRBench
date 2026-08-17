#!/usr/bin/env bash
#SBATCH --job-name=neural_lightrig
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=rtx_4090:1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=8G
#SBATCH --time=12:00:00
#SBATCH --output=logs/sbatch/neural_lightrig-%j.out
#SBATCH --error=logs/sbatch/neural_lightrig-%j.err

set -euo pipefail

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

readonly PROJECT_ROOT="/cluster/scratch/xiwang1/hiwi/PBREstimationEval"
readonly PYTHON="${PROJECT_ROOT}/third_party/.venvs/neural_lightrig/bin/python"

cd "${PROJECT_ROOT}"
mkdir -p logs/sbatch

if [[ ! -x "${PYTHON}" ]]; then
    echo "Neural LightRig environment not found: ${PYTHON}" >&2
    echo "Create it with: uv run python scripts/setup/neural_lightrig_deps.py" >&2
    exit 2
fi

exec "${PYTHON}" -u src/infer_pbr_2d.py \
    method_2d=neural_lightrig \
    data=all_2d \
    "$@"
