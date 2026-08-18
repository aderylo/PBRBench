#!/usr/bin/env bash
#SBATCH --job-name=preprocess_2d
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=rtx_4090:1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=8G
#SBATCH --time=12:00:00
#SBATCH --output=logs/sbatch/preprocess_2d-%j.out
#SBATCH --error=logs/sbatch/preprocess_2d-%j.err

set -euo pipefail

if [[ $# -lt 1 ]]; then
    echo "Usage: sbatch scripts/sbatch/preprocess_2d.sh <dataset_name>" >&2
    echo "Example: sbatch scripts/sbatch/preprocess_2d.sh texverse" >&2
    exit 1
fi

DATASET="$1"
shift

readonly PROJECT_ROOT="/cluster/scratch/xiwang1/hiwi/PBREstimationEval"
cd "${PROJECT_ROOT}"
mkdir -p logs/sbatch

module load blender/3.4.1

exec uv run python -u src/data/preprocessing/render_samples_2d.py \
    --config-name "data/preprocessing/render_${DATASET}_2d" \
    "$@"
