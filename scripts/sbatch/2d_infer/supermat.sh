#!/usr/bin/env bash
#SBATCH --job-name=supermat
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=rtx_4090:1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=8G
#SBATCH --time=12:00:00
#SBATCH --output=logs/sbatch/supermat-%j.out
#SBATCH --error=logs/sbatch/supermat-%j.err

set -euo pipefail

readonly PROJECT_ROOT="/cluster/scratch/xiwang1/hiwi/PBREstimationEval"
readonly PYTHON="${PROJECT_ROOT}/third_party/.venvs/supermat/bin/python"

cd "${PROJECT_ROOT}"
mkdir -p logs/sbatch

if [[ ! -x "${PYTHON}" ]]; then
    echo "SuperMat environment not found: ${PYTHON}" >&2
    echo "Create it with: uv run python scripts/setup/supermat_deps.py" >&2
    exit 2
fi

exec "${PYTHON}" -u src/infer_pbr_2d.py \
    method_2d=supermat \
    data=all_2d \
    "$@"
