#!/bin/bash
# GPU-memory test of the 60k-bin run with the GPU-resident streaming ring.
# One short round (epochs: 12) so the profiler sees training, callbacks,
# coverage guard and truncation. Submit from the repo root:
#   sbatch slurm/memtest_60k.sbatch mlp
#   sbatch slurm/memtest_60k.sbatch conv
#SBATCH --job-name=memtest_60k
#SBATCH --partition=boost_usr_prod
#SBATCH --account=Sis26_trotta
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=120G
#SBATCH --time=02:00:00
#SBATCH --output=slurm/logs/%x_%j.out
#SBATCH --error=slurm/logs/%x_%j.err

set -euo pipefail
VARIANT=${1:?usage: sbatch slurm/memtest_60k.sbatch mlp|conv}

cd "$SLURM_SUBMIT_DIR"
source activate_venv.sh
export TMPDIR=$SCRATCH/tmp   # Lightning ckpt saves go through TMPDIR
mkdir -p "$TMPDIR"

OBS=$PEMBHB_DATA_DIR/obs_maskzero_baseline_fmax1e-1_withnoise.h5

nvidia-smi --query-gpu=timestamp,utilization.gpu,memory.used --format=csv -l 20 \
    > slurm/logs/gpu_${SLURM_JOB_ID}_${VARIANT}.csv &

python scripts/tmnre_joint.py \
    --train-config train_60k_${VARIANT}_gpustream.yaml \
    --datagen-config datagen_maskzero_baseline_fmax1e-1.yaml \
    --obs-path "$OBS" \
    --n_rounds 1 \
    memtest60k_${VARIANT}
