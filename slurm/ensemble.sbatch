#!/bin/bash
# The seeds combined (mean probability) and flips, against the single best model: chosen on val, then test once.
#     DATA=~/new_dataset RUNS=~/deepcrack_runs/new sbatch slurm/ensemble.sbatch
# RUNS holds seed_*/best.pt (from train.sbatch --array). Writes RUNS/ensemble.json, which
# `predict --ensemble RUNS/ensemble.json` uses (its checkpoints, flips and val-chosen threshold).
# Optional: ENV_DIR (conda env, default ~/.conda/envs/swinunet).
#SBATCH --job-name=deepcrack_ens
#SBATCH --partition=ASL-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --output=deepcrack_ens_%j.out

set -uo pipefail
: "${DATA:?Set DATA=<dataset folder>}"
: "${RUNS:?Set RUNS=<folder with seed_*/best.pt>}"
cd "${SLURM_SUBMIT_DIR:-$(pwd)}" || exit 1

# Compute nodes have no conda: put the environment on PATH instead of activating it.
ENV_DIR="${ENV_DIR:-$HOME/.conda/envs/swinunet}"
export PATH="$ENV_DIR/bin:$PATH"
export LD_LIBRARY_PATH="$ENV_DIR/lib:${LD_LIBRARY_PATH:-}"
export TORCH_HOME="${TORCH_HOME:-$HOME/weights_cache/torch}"
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 MPLBACKEND=Agg

echo "===== deepcrack ensemble ${SLURM_JOB_ID:-} | $DATA | $RUNS | $(hostname) | $(date) ====="
echo "code   : $(git rev-parse --short HEAD 2>/dev/null)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo "no GPU visible"

python -u train_deepcrack.py ensemble --runs "$RUNS" --data "$DATA" --workers "${SLURM_CPUS_PER_TASK:-8}"
STATUS=$?
echo "===== finished $(date) | exit $STATUS ====="
exit $STATUS
