#!/usr/bin/env bash
#SBATCH --job-name=consensus-array
#SBATCH --account=def-razvan05
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --array=0-499   # <-- set after computing BINS*NBATCH-1
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.err

set -euo pipefail

# ------------- Tunables -------------
BINS=100                 # number of sigma bins in [sigma_min, sigma_max]
INIT_TOTAL=100       # total initializations
RUNS_PER_INIT=50       # per initialization, how many runs to average
INITS_PER_TASK=20       # batch size -> NBATCH = INIT_TOTAL / INITS_PER_TASK
SIGMA_MIN=0.0
SIGMA_MAX=0.5

N=32; D=3; BETA=5.0
TMAX=5000.0; DT=0.01
THRESHOLD=1e-2
STORE_STRIDE=50
MEAN_UPDATE_STRIDE=5

JOBNAME=consensus_sweep
OUTDIR=results
PY=python

# ------------- Deriving array size
NBATCH=$(( INIT_TOTAL / INITS_PER_TASK ))
if [ $(( INIT_TOTAL % INITS_PER_TASK )) -ne 0 ]; then
  echo "INIT_TOTAL must be divisible by INITS_PER_TASK" >&2
  exit 1
fi
TOTAL=$(( BINS * NBATCH ))

# Map array index -> (sigma_idx, batch_idx)
AID=${SLURM_ARRAY_TASK_ID:-0}
SIG_IDX=$(( AID / NBATCH ))
BATCH_IDX=$(( AID % NBATCH ))
INIT_START=$(( BATCH_IDX * INITS_PER_TASK ))

echo "[info] AID=$AID -> sigma_idx=$SIG_IDX, init_start=$INIT_START, inits_per_task=$INITS_PER_TASK"

# Make dirs
mkdir -p "$OUTDIR" logs

# Activate venv 
source ~/venvs/jaxcc/bin/activate

# optional: pin moderate threading for JAX/XLA on CPU
export XLA_FLAGS="--xla_cpu_multi_thread_eigen=true intra_op_parallelism_threads=4 inter_op_parallelism_threads=1"
export OMP_NUM_THREADS=4

$PY consensus_time_vs_sigma_hpc.py \
  --n "$N" --d "$D" --b "$BETA" \
  --Tmax "$TMAX" --dt "$DT" \
  --threshold "$THRESHOLD" \
  --sigma_min "$SIGMA_MIN" --sigma_max "$SIGMA_MAX" --bins "$BINS" \
  --runs_per_init "$RUNS_PER_INIT" \
  --inits_per_task "$INITS_PER_TASK" \
  --init_start "$INIT_START" \
  --store_stride "$STORE_STRIDE" \
  --mean_update_stride "$MEAN_UPDATE_STRIDE" \
  --sigma_idx "$SIG_IDX" \
  --outdir "$OUTDIR" \
  --jobname "$JOBNAME"
