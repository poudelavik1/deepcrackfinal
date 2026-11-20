#!/bin/bash
# Score finished runs on a split with this code (the threshold stored in best.pt, chosen on val). From the repo folder:
#     DATA=~/new_dataset RUNS=~/deepcrack_runs/new sbatch slurm/evaluate.sbatch
# Every RUNS/seed_*/best.pt (or RUNS/best.pt) is scored and written to <run>/<split>_eval.json; for the test split
# the seeds are then summarized (mean +- std, the run to keep) in RUNS/summary_test_eval.{csv,json}.
# Optional: SPLIT (default test), ENV_DIR (conda env, default ~/.conda/envs/swinunet).
#SBATCH --job-name=deepcrack_eval
#SBATCH --partition=ASL-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=deepcrack_eval_%j.out

set -uo pipefail
: "${DATA:?Set DATA=<dataset folder>}"
: "${RUNS:?Set RUNS=<folder of the run, or of its seed_* runs>}"
SPLIT="${SPLIT:-test}"
cd "${SLURM_SUBMIT_DIR:-$(pwd)}" || exit 1

# Compute nodes have no conda: put the environment on PATH instead of activating it.
ENV_DIR="${ENV_DIR:-$HOME/.conda/envs/swinunet}"
export PATH="$ENV_DIR/bin:$PATH"
export LD_LIBRARY_PATH="$ENV_DIR/lib:${LD_LIBRARY_PATH:-}"
export TORCH_HOME="${TORCH_HOME:-$HOME/weights_cache/torch}"
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 MPLBACKEND=Agg

echo "===== deepcrack evaluation ${SLURM_JOB_ID:-} | $SPLIT split of $DATA | $RUNS | $(hostname) | $(date) ====="
echo "code   : $(git rev-parse --short HEAD 2>/dev/null)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo "no GPU visible"

STATUS=0
COUNT=0
for CHECKPOINT in "$RUNS"/seed_*/best.pt "$RUNS"/best.pt; do
    [ -f "$CHECKPOINT" ] || continue
    RUN="$(dirname "$CHECKPOINT")"
    echo "----- $RUN"
    python -u train_deepcrack.py evaluate --checkpoint "$CHECKPOINT" --data "$DATA" --split "$SPLIT" \
        --workers "${SLURM_CPUS_PER_TASK:-8}" --save "$RUN/${SPLIT}_eval.json" || STATUS=1
    COUNT=$((COUNT + 1))
done
[ "$COUNT" -gt 0 ] || { echo "No best.pt under $RUNS" >&2; exit 1; }
if [ "$SPLIT" = "test" ] && [ "$STATUS" -eq 0 ] && ls "$RUNS"/seed_*/best.pt > /dev/null 2>&1; then
    python -u train_deepcrack.py summarize "$RUNS" --test-file test_eval.json || STATUS=1
fi
echo "===== finished $(date) | exit $STATUS ====="
exit $STATUS
