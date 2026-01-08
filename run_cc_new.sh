#!/bin/bash
# CPU version: 1 JAX process per job, multithreaded via cpus-per-task.
# Call with: sbatch run_cc.sh

#SBATCH --job-name=cc_sweep_cpu
#SBATCH --account=def-razvan05
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8           # try 4 or 8; 8 is a good start on Fir
#SBATCH --mem=64G
#SBATCH --time=0-06:00
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
# NOTE: we do NOT set --array here; the script will submit the array itself.

set -euo pipefail

# ------------------- user parameters -------------------
NUM_SIGMA=100
SIGMA_MIN=0.0
SIGMA_MAX=1.0

TOT_INITS=100             # total init sets across all shards
INITS_PER_TASK=5          # inits per shard
RUNS_PER_INIT=100         # runs per init

N=32
D=3
BETA=2.0
DT=0.001
TMAX=500.0
THRESH=1e-2

MEAN_UPDATE_STRIDE=1
MEAN_REFINE_STEPS=2

INIT_SEED=123
NOISE_SEED=999
OUTDIR=out_cc_sweep_cpu
JOBNAME=cc_sweep_cpu
PY=python

mkdir -p logs "${OUTDIR}"

# ------------------- derived quantities -------------------
NUM_SHARDS=$(( (TOT_INITS + INITS_PER_TASK - 1) / INITS_PER_TASK ))
TOTAL=$(( NUM_SIGMA * NUM_SHARDS ))      # total (sigma_idx, shard_idx) pairs

# ------------------- auto-submit array if needed -------------------
# If we're not already inside an array job, submit the full array and exit.
if [[ -z "${SLURM_ARRAY_TASK_ID:-}" ]]; then
    echo "[info] Launching CPU sweep..."
    echo "[info] NUM_SIGMA=${NUM_SIGMA}, NUM_SHARDS=${NUM_SHARDS}, TOTAL=${TOTAL}"
    sbatch --array=0-$((TOTAL-1)) "$0"
    exit 0
fi

# ------------------- inside a running array task -------------------
TASK=${SLURM_ARRAY_TASK_ID}

if (( TASK < 0 || TASK >= TOTAL )); then
    echo "[ERROR] TASK=${TASK} out of range [0, ${TOTAL})"
    exit 1
fi

SIGMA_IDX=$(( TASK % NUM_SIGMA ))
SHARD_IDX=$(( TASK / NUM_SIGMA ))

echo "[info] TOTAL=${TOTAL} (NUM_SIGMA=${NUM_SIGMA}, NUM_SHARDS=${NUM_SHARDS})"
echo "[info] TASK=${TASK}, sigma_idx=${SIGMA_IDX}, shard_idx=${SHARD_IDX}"

# ------------------- threading + JAX CPU env -------------------
# Use all cpus-per-task threads for JAX / BLAS.
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export XLA_FLAGS="--xla_cpu_multi_thread_eigen=true intra_op_parallelism_threads=${SLURM_CPUS_PER_TASK:-1}"

# ------------------- run the worker -------------------
$PY consensus_time_vs_sigma_hpc.py \
  --n ${N} --d ${D} --b ${BETA} --dt ${DT} --Tmax ${TMAX} \
  --threshold ${THRESH} \
  --sigma_min ${SIGMA_MIN} --sigma_max ${SIGMA_MAX} --bins ${NUM_SIGMA} \
  --sigma_idx ${SIGMA_IDX} \
  --tot_inits ${TOT_INITS} --runs_per_init ${RUNS_PER_INIT} \
  --inits_per_task ${INITS_PER_TASK} --shard_idx ${SHARD_IDX} \
  --mean_update_stride ${MEAN_UPDATE_STRIDE} \
  --mean_refine_steps ${MEAN_REFINE_STEPS} \
  --init_seed ${INIT_SEED} --noise_seed ${NOISE_SEED} \
  --outdir "${OUTDIR}" --jobname "${JOBNAME}"
