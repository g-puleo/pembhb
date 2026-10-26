#!/bin/bash
# Heterodyned 84-d test run on Leonardo Booster: 1 round, 10-sigma Fisher prior, 300 epochs.
# Submit from the repo root (the job runs there):
#   sbatch slurm/het84d_fisher10.sbatch                                         # 24 h, normal QOS
#   sbatch --qos=boost_qos_dbg --time=00:30:00 slurm/het84d_fisher10.sbatch     # 30-min benchmark
# Needs, under $PEMBHB_DATA_DIR/long_tobs_heterodyne/het8000/: het_8000.npz, obs_84d_het8000_withnoise.h5
# Outputs: slurm/logs/<name>_<jobid>.{out,err} (Slurm), slurm/logs/<name>_<jobid>/ (gpu.csv, spin/mask logs, summary)

#SBATCH --job-name=het84d_fisher10
#SBATCH --account=Sis26_trotta
#SBATCH --partition=boost_usr_prod
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=120G
#SBATCH --output=slurm/logs/%x_%j.out
#SBATCH --error=slurm/logs/%x_%j.err

REPO=$SLURM_SUBMIT_DIR
cd "$REPO"
source "$REPO/activate_venv.sh"   # modules + venv + PEMBHB_* dirs
set -uo pipefail

RUN_NAME=het84d_fisher10_j${SLURM_JOB_ID}
JOBDIR=$REPO/slurm/logs/${SLURM_JOB_NAME}_${SLURM_JOB_ID}
HETDIR=$PEMBHB_DATA_DIR/long_tobs_heterodyne/het8000
OBS=$HETDIR/obs_84d_het8000_withnoise.h5
TRAIN_CONFIG=train_het84d_fisher10.yaml
DATAGEN_CONFIG=datagen_maskzero_rt_84d_het.yaml
mkdir -p "$JOBDIR" "$PEMBHB_DATA_DIR/tmp"
export TMPDIR=$PEMBHB_DATA_DIR/tmp   # checkpoint saves stage through TMPDIR

# --- what ran, where ----------------------------------------------------------
echo "=== job $SLURM_JOB_ID on $(hostname) at $(date -Is) ==="
echo "run name : $RUN_NAME"
echo "git      : $(git rev-parse --short HEAD)$(git diff --quiet HEAD -- src scripts configs || echo ' +uncommitted changes')"
echo "python   : $(command -v python)"
echo "data dir : $PEMBHB_DATA_DIR"
echo "plots dir: $PEMBHB_PLOTS_DIR"
echo "obs      : $OBS"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda, 'gpu ok', torch.cuda.is_available())"
cp "configs/$TRAIN_CONFIG" "configs/$DATAGEN_CONFIG" "$JOBDIR/"

# --- fail early, with a readable reason --------------------------------------
for f in "$OBS" "$HETDIR/het_8000.npz"; do
    [[ -f $f ]] || { echo "MISSING INPUT: $f" >&2; exit 2; }
done
python -c "import pembhb.heterodyne" || { echo "pembhb.heterodyne not importable: pull commit 7715628 or later" >&2; exit 2; }

# --- GPU usage every 20 s -----------------------------------------------------
nvidia-smi --query-gpu=timestamp,utilization.gpu,memory.used,memory.total --format=csv -l 20 > "$JOBDIR/gpu.csv" &
SMI=$!

# --- the run ------------------------------------------------------------------
# stdout: spin / mask-rejection chatter goes to its own files; tracebacks go to the .err file
python -u scripts/tmnre_joint.py \
    --train-config "$TRAIN_CONFIG" \
    --datagen-config "$DATAGEN_CONFIG" \
    --obs-path "$OBS" \
    --n_rounds 1 \
    "$RUN_NAME" \
  | awk -v spin="$JOBDIR/spin.log" -v mask="$JOBDIR/mask.log" '
        /spin rejection/      { print >> spin; next }
        /\[MaskRejectSampler/ { print >> mask; next }
        { print; fflush() }'
status=${PIPESTATUS[0]}

kill $SMI 2>/dev/null

# --- summary ------------------------------------------------------------------
{
    echo "=== exit status $status at $(date -Is) (elapsed ${SECONDS}s) ==="
    echo "peak GPU memory: $(awk -F', ' 'NR>1 {gsub(/ MiB/,"",$3); if ($3>m) m=$3} END {print m+0}' "$JOBDIR/gpu.csv") MiB"
    echo "last epoch line: $(grep 'Joint | epoch' "$REPO/slurm/logs/${SLURM_JOB_NAME}_${SLURM_JOB_ID}.out" | tail -1)"
    if [[ $status -ne 0 ]]; then
        echo "--- last 30 lines of stderr ---"
        tail -n 30 "$REPO/slurm/logs/${SLURM_JOB_NAME}_${SLURM_JOB_ID}.err"
    fi
} | tee "$JOBDIR/summary.txt"
exit $status
