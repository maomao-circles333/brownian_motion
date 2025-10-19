#!/bin/bash
#SBATCH --job-name=cc_drift
#SBATCH --account=def-razvan05 
#SBATCH --time=06:00:00
#SBATCH --cpus-per-task=64
#SBATCH --mem=64G
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#SBATCH --array=0-499%200             # 150 sigma-bins × 20 shards/bin

set -euo pipefail
set -x

# Sweep & sharding
BINS=50
SIGMA_MIN=0.0
SIGMA_MAX=1.5

TOT_INITS=50
RUNS_PER_INIT=50
INITS_PER_TASK=5
NUM_SHARDS=$(( (TOT_INITS + INITS_PER_TASK - 1) / INITS_PER_TASK ))  # 20 shards/bin

# Dynamics
N=32
D=3
BETA=2.0
DT=0.001
TMAX=300.0
THRESH=0.01
 
# Intrinsic-mean params
STORE_STRIDE=20
MEAN_UPDATE_STRIDE=20
MEAN_REFINE_STEPS=2

PROJECT_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
OUTDIR="${PROJECT_DIR}/out_cc_drift"
JOBNAME="cc_drift"
INIT_SEED=123
NOISE_SEED=999

cd "$PROJECT_DIR"
mkdir -p "$OUTDIR" logs

# module --force purge
module load StdEnv/2023
(module load python/3.11 2>/dev/null) || (module load python/3.10 2>/dev/null) || module load python
python -V || { echo "ERROR: Python module missing"; exit 1; }

VENV_DIR="${SCRATCH}/venvs/ccsweep"
if [ ! -d "$VENV_DIR" ]; then
  python -m venv "$VENV_DIR"
  source "$VENV_DIR/bin/activate"
  pip install --upgrade pip wheel
  pip install --no-cache-dir "jax[cpu]" numpy matplotlib
else
  source "$VENV_DIR/bin/activate"
fi

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export XLA_FLAGS="--xla_cpu_multi_thread_eigen=true intra_op_parallelism_threads=${SLURM_CPUS_PER_TASK:-1}"
export JAX_PLATFORM_NAME=cpu
export JAX_ENABLE_X64=0
export MPLBACKEND=Agg

GLOBAL_ID=${SLURM_ARRAY_TASK_ID}
SIGMA_IDX=$(( GLOBAL_ID % BINS ))
SHARD_IDX=$(( GLOBAL_ID / BINS ))

echo "[INFO] BINS=$BINS NUM_SHARDS=$NUM_SHARDS ARRAY_SIZE=$((BINS*NUM_SHARDS))"
echo "[INFO] GLOBAL_ID=$GLOBAL_ID  SIGMA_IDX=$SIGMA_IDX  SHARD_IDX=$SHARD_IDX"

srun python consensus_drift_vs_sigma_hpc.py \
  --n ${N} --d ${D} --b ${BETA} \
  --dt ${DT} --Tmax ${TMAX} --threshold ${THRESH} \
  --sigma_min ${SIGMA_MIN} --sigma_max ${SIGMA_MAX} --bins ${BINS} --sigma_idx ${SIGMA_IDX} \
  --tot_inits ${TOT_INITS} --runs_per_init ${RUNS_PER_INIT} \
  --inits_per_task ${INITS_PER_TASK} --shard_idx ${SHARD_IDX} \
  --init_seed ${INIT_SEED} --noise_seed ${NOISE_SEED} \
  --store_stride ${STORE_STRIDE} --mean_update_stride ${MEAN_UPDATE_STRIDE} --mean_refine_steps ${MEAN_REFINE_STEPS} \
  --outdir "${OUTDIR}" --jobname "${JOBNAME}"
