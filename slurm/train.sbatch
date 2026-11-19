#!/bin/bash
# Train DeepCrack with the final recipe on the ASL cluster. From the repo folder:
#     DATA=~/new_dataset OUT=~/deepcrack_runs/new_s0 SEED=0 sbatch slurm/train.sbatch
# Optional: ENV_DIR (conda env, default ~/.conda/envs/swinunet), EXTRA_ARGS (e.g. "--recipe benchmark").
# Resubmitting the same command after a time-out resumes the run from last.pt.
#SBATCH --job-name=deepcrack
#SBATCH --partition=ASL-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=deepcrack_%j.out

set -uo pipefail
: "${DATA:?Set DATA=<dataset folder>}"
: "${OUT:?Set OUT=<run folder>}"
cd "${SLURM_SUBMIT_DIR:-$(pwd)}" || exit 1

# Compute nodes have no conda: put the environment on PATH instead of activating it.
ENV_DIR="${ENV_DIR:-$HOME/.conda/envs/swinunet}"
export PATH="$ENV_DIR/bin:$PATH"
export LD_LIBRARY_PATH="$ENV_DIR/lib:${LD_LIBRARY_PATH:-}"
export TORCH_HOME="${TORCH_HOME:-$HOME/weights_cache/torch}"   # VGG-16-BN weights, downloaded on the login node
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 MPLBACKEND=Agg

echo "===== deepcrack ${SLURM_JOB_ID:-} on $(hostname) | $(date) ====="
echo "code   : $(git rev-parse --short HEAD 2>/dev/null)"
echo "python : $(command -v python)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo "no GPU visible"

# In the batch step, not srun: an srun step ignores --cpus-per-task.
python -u train_deepcrack.py train --data "$DATA" --out "$OUT" --seed "${SEED:-0}" \
    --workers "${SLURM_CPUS_PER_TASK:-8}" ${EXTRA_ARGS:-}
STATUS=$?
echo "===== finished $(date) | exit $STATUS ====="
exit $STATUS
