#!/usr/bin/env bash
#SBATCH --job-name=direct_eval_2d
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=4G
#SBATCH --time=01:00:00
#SBATCH --output=logs/sbatch/direct_eval_2d-%j.out
#SBATCH --error=logs/sbatch/direct_eval_2d-%j.err

# Submit from the repository root:
#   sbatch scripts/sbatch/direct_eval_2d.sh <predictions_dir> [extra Hydra options...]
# Example:
#   sbatch scripts/sbatch/direct_eval_2d.sh outputs/pbr_2d/supermat/predictions

set -euo pipefail

if [[ $# -lt 1 ]]; then
    echo "Usage: sbatch scripts/sbatch/direct_eval_2d.sh <predictions_dir> [hydra options...]" >&2
    echo "Example: sbatch scripts/sbatch/direct_eval_2d.sh outputs/pbr_2d/supermat/predictions" >&2
    exit 1
fi

readonly PREDICTIONS_DIR="$1"
shift

readonly PROJECT_ROOT="/cluster/scratch/xiwang1/hiwi/PBREstimationEval"

cd "${PROJECT_ROOT}"
mkdir -p logs/sbatch

exec uv run python -u src/eval_pbr_2d_direct.py \
    predictions_dir="${PREDICTIONS_DIR}" \
    "$@"
