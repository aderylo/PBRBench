#!/usr/bin/env bash
#SBATCH --job-name=indirect_eval_2d
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=rtx_4090:1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=8G
#SBATCH --time=04:00:00
#SBATCH --output=logs/sbatch/indirect_eval_2d-%j.out
#SBATCH --error=logs/sbatch/indirect_eval_2d-%j.err

# Submit from the repository root:
#   sbatch scripts/sbatch/indirect_eval_2d.sh <predictions_dir> [extra Hydra options...]
# Example:
#   sbatch scripts/sbatch/indirect_eval_2d.sh outputs/pbr_2d/diffusion_renderer/predictions

set -euo pipefail

if [[ $# -lt 1 ]]; then
    echo "Usage: sbatch scripts/sbatch/indirect_eval_2d.sh <predictions_dir> [hydra options...]" >&2
    echo "Example: sbatch scripts/sbatch/indirect_eval_2d.sh outputs/pbr_2d/diffusion_renderer/predictions" >&2
    exit 1
fi

readonly PREDICTIONS_DIR="$1"
shift

readonly PROJECT_ROOT="/cluster/scratch/xiwang1/hiwi/PBREstimationEval"

cd "${PROJECT_ROOT}"
mkdir -p logs/sbatch

module load blender/3.4.1

if ! command -v blender >/dev/null 2>&1; then
    echo "Blender is unavailable after 'module load blender'." >&2
    exit 2
fi

exec uv run python -u src/rerendering_eval_pbr_2d.py \
    predictions_dir="${PREDICTIONS_DIR}" \
    save_rerenders=true \
    "$@"
