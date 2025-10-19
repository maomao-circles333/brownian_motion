#!/bin/bash
# Compute Canada SLURM array launcher for consensus_time_vs_sigma_hpc.py

#SBATCH --job-name=cc_sweep
#SBATCH --account=def-razvan05
#SBATCH --cpus-per-task=64
#SBATCH --mem=64G
#SBATCH --time=0-06:00             # 3 hours
#SBATCH --output=logs/%x_%A_%a.out # logs/cc_sweep_JOBID_TASKID.out
#SBATCH --error=logs/%x_%A_%a.err
# Array size is set at runtime below with sbatch --array

set -euo pipefail

NUM_SIGMA=50             # bins: sigma in [0.0, 0.5]
SIGMA_MIN=0.0
SIGMA_MAX=1.0
TOT_INITS=100            # total number of init sets across all shards
INITS_PER_TASK=5         # how many inits per shard
RUNS_PER_INIT=100         # runs per init (R)
N=32
D=3
BETA=2.0
DT=0.001
TMAX=500.0
THRESH=1e-2
STORE_STRIDE=10           # storage cadence; does NOT affect dynamics
MEAN_UPDATE_STRIDE=1      # if mean used in dynamics, keep at 1
MEAN_REFINE_STEPS=2
INIT_SEED=123
NOISE_SEED=999
OUTDIR=out_cc_sweep
JOBNAME=cc_sweep
PY=python                 

# ------------------- derived -------------------
mkdir -p logs "${OUTDIR}"

NUM_SHARDS=$(( (TOT_INITS + INITS_PER_TASK - 1) / INITS_PER_TASK ))

# If running this file directly via bash, how to submit:
if [[ -z "${SLURM_ARRAY_TASK_ID:-}" ]]; then
  TOTAL=$(( NUM_SIGMA * NUM_SHARDS ))
  echo "Submitting ${TOTAL} tasks (${NUM_SIGMA} sigmas × ${NUM_SHARDS} shards) ..."
  sbatch --array=0-$((TOTAL-1)) "$0"
  exit 0
fi

TASK=${SLURM_ARRAY_TASK_ID}

SIGMA_IDX=$(( TASK % NUM_SIGMA ))
SHARD_IDX=$(( TASK / NUM_SIGMA ))

echo "[info] TASK=${TASK}  sigma_idx=${SIGMA_IDX}  shard_idx=${SHARD_IDX}"

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
  --outdir ${OUTDIR} --jobname ${JOBNAME}
