#!/usr/bin/env bash
#SBATCH --job-name=pbr_oracle
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=rtx_4090:1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=8G
#SBATCH --time=12:00:00
#SBATCH --output=logs/sbatch/pbr_oracle-%j.out
#SBATCH --error=logs/sbatch/pbr_oracle-%j.err

set -euo pipefail

readonly PROJECT_ROOT="/cluster/scratch/xiwang1/hiwi/PBREstimationEval"

cd "${PROJECT_ROOT}"
mkdir -p logs/sbatch

module load blender/3.4.1

if ! command -v blender >/dev/null 2>&1; then
    echo "Blender is unavailable after 'module load blender'." >&2
    exit 2
fi

exec uv run python -u src/infer_pbr_2d.py \
    method_2d=pbr_oracle \
    data=all_2d \
    "$@"
