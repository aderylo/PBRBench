#!/usr/bin/env bash
#SBATCH --job-name=diffusion_renderer
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=rtx_4090:1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=8G
#SBATCH --time=12:00:00
#SBATCH --output=logs/sbatch/diffusion_renderer-%j.out
#SBATCH --error=logs/sbatch/diffusion_renderer-%j.err

set -euo pipefail

readonly PROJECT_ROOT="/cluster/scratch/xiwang1/hiwi/PBREstimationEval"
readonly PYTHON="${PROJECT_ROOT}/third_party/.venvs/diffusion_renderer/bin/python"

cd "${PROJECT_ROOT}"
mkdir -p logs/sbatch

if [[ ! -x "${PYTHON}" ]]; then
    echo "DiffusionRenderer environment not found: ${PYTHON}" >&2
    echo "Create it with: uv run python scripts/setup/diffusion_renderer_deps.py" >&2
    exit 2
fi

exec "${PYTHON}" -u src/infer_pbr_2d.py \
    method_2d=diffusion_renderer \
    data=all_2d \
    "$@"
