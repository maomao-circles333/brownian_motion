#!/bin/bash
#SBATCH --job-name=cp_sweep_d
#SBATCH --account=def-razvan05
#SBATCH --time=10:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=8G
#SBATCH --array=0-236
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err

# Usage:
#   sbatch --export=ALL,DIM=10 run_cp_sweep.sh
#
# This script runs one DIM at a time.
# Array index is sigma_idx (0..bins-1). also shard over inits manually.

set -euo pipefail

module load python/3.11 2>/dev/null || true

DIM=${DIM:?Please pass DIM via --export=ALL,DIM=...}

N=25
BETA=1.0
DT=0.01
TMAX=500.0
THRESH=0.001

# Sweep params
SIGMA_MIN=0.0
SIGMA_MAX=2.0
BINS=237
SIGMA_IDX=${SLURM_ARRAY_TASK_ID}

# for averaging
RUNS_MEAN=100   # runs per init (mean trajectory)
RUN_CP=100       # number of inits (# mean trajectories)

# Sharding over inits:
# Choose inits_per_task so total shards = ceil(RUN_CP / inits_per_task).
# We launch multiple shards per sigma_idx by looping here.
INITS_PER_TASK=5
NUM_SHARDS=$(( (RUN_CP + INITS_PER_TASK - 1) / INITS_PER_TASK ))

OUTROOT="out_cp_sweep"
JOBNAME="cp_sweep"

mkdir -p logs
module load python/3.11
source ~/venvs/cp_meantraj/bin/activate
python -c "import jax; print('JAX OK')"

for SHARD_IDX in $(seq 0 $((NUM_SHARDS - 1))); do
  srun --exclusive -N 1 -n 1 \
    python3 cc_meantraj_cp_sweep.py \
      --n ${N} --d ${DIM} --beta ${BETA} \
      --dt ${DT} --Tmax ${TMAX} --threshold ${THRESH} \
      --sigma_min ${SIGMA_MIN} --sigma_max ${SIGMA_MAX} --bins ${BINS} --sigma_idx ${SIGMA_IDX} \
      --runs_mean ${RUNS_MEAN} --run_cp ${RUN_CP} \
      --inits_per_task ${INITS_PER_TASK} --shard_idx ${SHARD_IDX} \
      --outroot ${OUTROOT} --jobname ${JOBNAME}
done

