#!/bin/bash
#SBATCH --job-name=beta_sweep_d
#SBATCH --account=def-razvan05
#SBATCH --output=logs/beta_sweep_d_%A_%a.out
#SBATCH --error=logs/beta_sweep_d_%A_%a.err
#SBATCH --time=06:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=8
#SBATCH --array=0-59

set -euo pipefail

module load python/3.11
source ~/venvs/cp_meantraj/bin/activate

if [[ -z "${DIM:-}" ]]; then
  echo "ERROR: DIM not set. Submit with: sbatch --export=ALL,DIM=<d>,SIGMA0=<sigma> run_beta_sweep.sh"
  exit 1
fi
if [[ -z "${SIGMA0:-}" ]]; then
  echo "ERROR: SIGMA0 not set. Submit with: sbatch --export=ALL,DIM=<d>,SIGMA0=<sigma> run_beta_sweep.sh"
  exit 1
fi

N=100
DT=0.01
TMAX=500
THRESH=0.001

RUN_CP=100
RUNS_PER_INIT=100
INITS_PER_TASK=5

CHECK_STRIDE=10
MEAN_REFINE_STEPS=2

BETA_MIN=0.0
BETA_MAX=8.0
BETA_BINS=60
BETA_BREAK=1.0
BETA_LIN_BINS=21
# shard indexing: 0..(ceil(RUN_CP/INITS_PER_TASK)-1)
NUM_SHARDS=$(( (RUN_CP + INITS_PER_TASK - 1) / INITS_PER_TASK ))

mkdir -p logs

# loop over shards inside each beta array task
for SHARD in $(seq 0 $((NUM_SHARDS-1))); do
  python cc_beta_sweep_cp.py \
    --d "${DIM}" --n "${N}" \
    --dt "${DT}" --Tmax "${TMAX}" --threshold "${THRESH}" \
    --sigma0 "${SIGMA0}" \
    --beta_min "${BETA_MIN}" --beta_max "${BETA_MAX}" --beta_bins "${BETA_BINS}" \
    --beta_break "${BETA_BREAK}" --beta_lin_bins "${BETA_LIN_BINS}" \
    --beta_idx "${SLURM_ARRAY_TASK_ID}" \
    --run_cp "${RUN_CP}" --runs_per_init "${RUNS_PER_INIT}" \
    --inits_per_task "${INITS_PER_TASK}" --shard_idx "${SHARD}" \
    --check_stride "${CHECK_STRIDE}" --mean_refine_steps "${MEAN_REFINE_STEPS}" \
    --outroot out_cp_beta_sweep --jobname beta_sweep
done
