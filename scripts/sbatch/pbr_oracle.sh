#!/usr/bin/env bash
#SBATCH --job-name=pbr_oracle
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=rtx_4090:1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=8G
#SBATCH --time=12:00:00
#SBATCH --output=pbr_oracle-%j.out
#SBATCH --error=pbr_oracle-%j.err

# Submit from the repository root:
#   sbatch scripts/sbatch/pbr_oracle.sh
#
# Add Hydra overrides after the script path when needed, for example:
#   sbatch scripts/sbatch/pbr_oracle.sh data=polyhaven_2d

set -euo pipefail

readonly PROJECT_ROOT="/cluster/scratch/xiwang1/hiwi/PBREstimationEval"

cd "${PROJECT_ROOT}"

module load blender/3.4.1

if ! command -v blender >/dev/null 2>&1; then
    echo "Blender is unavailable after 'module load blender'." >&2
    exit 2
fi

exec uv run python -u src/infer_pbr_2d.py \
    method_2d=pbr_oracle \
    "$@"
