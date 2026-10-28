#!/bin/bash
# Production TMNRE run on Leonardo Booster: conv encoder, 60k bins (fmax 0.1),
# GPU-resident streaming ring. Argument: the simulator,
#   fd (default)  plain bbhx FD, the mlp60k observation        -> run conv60k_gpustream
#   rt            round trip (77 d lookback, 1-d Tukey taper),
#                 obs_nested_rt/obs_7d_withnoise.h5             -> run conv60k_rt_gpustream
# e.g. sbatch -J conv60k_rt slurm/conv60k.sbatch rt   (pass the same argument to every job of a chain).
# Normal QOS caps a job at 24 h, so queue a chain of jobs; each one picks up the
# run where the last stopped. Submit from the repo root, e.g. 5 chained jobs:
#   jid=$(sbatch --parsable slurm/conv60k.sbatch)
#   for i in 2 3 4 5; do jid=$(sbatch --parsable --dependency=afterany:$jid slurm/conv60k.sbatch); done
# or 4-day jobs on the long-production QOS (same chaining):
#   LP="--qos=boost_qos_lprod --time=4-00:00:00"
#   jid=$(sbatch --parsable $LP slurm/conv60k.sbatch)
#   jid=$(sbatch --parsable $LP --dependency=afterany:$jid slurm/conv60k.sbatch)
# Each job finds the run by RUN_NAME, moves aside the round the previous job left
# unfinished, and resumes for the rounds still missing up to MAX_ROUNDS. It exits
# at once when the chain has converged or MAX_ROUNDS are done, and with an error
# when the coverage guard rejected a truncation (inspect it, then resubmit).
# Needs, under $PEMBHB_DATA_DIR: obs_maskzero_baseline_fmax1e-1_withnoise.{h5,yaml} (fd)
# or obs_nested_rt/obs_7d_withnoise.{h5,yaml} (rt; bbhx needs the tstart patch)
# Outputs: slurm/logs/<name>_<jobid>.{out,err} (Slurm), slurm/logs/<name>_<jobid>/ (gpu.csv, spin/mask logs, summary)

#SBATCH --job-name=conv60k
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

MAX_ROUNDS=80
TRAIN_CONFIG=train_conv60k_gpustream.yaml
SIM=${1:-fd}
case $SIM in
    fd) RUN_NAME=conv60k_gpustream
        DATAGEN_CONFIG=datagen_maskzero_baseline_fmax1e-1.yaml
        OBS=$PEMBHB_DATA_DIR/obs_maskzero_baseline_fmax1e-1_withnoise.h5 ;;
    rt) RUN_NAME=conv60k_rt_gpustream
        DATAGEN_CONFIG=datagen_maskzero_rt_1w.yaml
        OBS=$PEMBHB_DATA_DIR/obs_nested_rt/obs_7d_withnoise.h5 ;;
    *)  echo "usage: sbatch slurm/conv60k.sbatch [fd|rt]" >&2; exit 2 ;;
esac
JOBDIR=$REPO/slurm/logs/${SLURM_JOB_NAME}_${SLURM_JOB_ID}
mkdir -p "$JOBDIR" "$PEMBHB_DATA_DIR/tmp"
export TMPDIR=$PEMBHB_DATA_DIR/tmp   # checkpoint saves stage through TMPDIR

for f in "$OBS" "${OBS%.h5}.yaml" "configs/$TRAIN_CONFIG" "configs/$DATAGEN_CONFIG"; do
    [[ -f $f ]] || { echo "MISSING INPUT: $f" >&2; exit 2; }
done

# --- where the run stands -----------------------------------------------------
TAGS=($(cd "$PEMBHB_DATA_DIR/logs" 2>/dev/null && ls -d 20*/*/*/channelizedmlp_${RUN_NAME} 2>/dev/null))
(( ${#TAGS[@]} <= 1 )) || { echo "several runs named $RUN_NAME: ${TAGS[*]}" >&2; exit 2; }
TAG=${TAGS[0]:-}
DONE=0
if [[ -n $TAG ]]; then
    D=$PEMBHB_DATA_DIR/$TAG
    while [[ -f $D/prior_after_round_$((DONE + 1)).yaml ]]; do DONE=$((DONE + 1)); done
    NEXT=$((DONE + 1))
    if grep -qs "STOP chain" "$D/chain_convergence.log"; then
        echo "$TAG: chain converged after $DONE rounds, nothing to do"; exit 0; fi
    (( DONE < MAX_ROUNDS )) || { echo "$TAG: $DONE rounds done, nothing to do"; exit 0; }
    if compgen -G "$D/rejected_truncation_round_${NEXT}*" > /dev/null; then
        echo "$TAG: the coverage guard rejected round $NEXT (see $D/rejected_truncation_round_${NEXT}*)" >&2; exit 3; fi
    (( DONE > 0 )) || { echo "$TAG: round 1 never finished; move the run aside by hand" >&2; exit 3; }
    L=$PEMBHB_DATA_DIR/logs/$TAG/round_$NEXT
    if [[ -d $L ]]; then mv "$L" "$L.partial_job$SLURM_JOB_ID"; echo "moved unfinished $L aside"; fi
    ARGS=(--resume "$TAG" --n_rounds $((MAX_ROUNDS - DONE)))
else
    ARGS=(--n_rounds "$MAX_ROUNDS")
fi

# --- what ran, where ----------------------------------------------------------
echo "=== job $SLURM_JOB_ID on $(hostname) at $(date -Is) ==="
echo "run      : ${TAG:-new run '$RUN_NAME'} (rounds done: $DONE / $MAX_ROUNDS)"
echo "git      : $(git rev-parse --short HEAD)$(git diff --quiet HEAD -- src scripts configs || echo ' +uncommitted changes')"
echo "data dir : $PEMBHB_DATA_DIR"
echo "plots dir: $PEMBHB_PLOTS_DIR"
echo "obs      : $OBS"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
cp "configs/$TRAIN_CONFIG" "configs/$DATAGEN_CONFIG" "$JOBDIR/"

# --- GPU usage every 60 s -----------------------------------------------------
nvidia-smi --query-gpu=timestamp,utilization.gpu,memory.used,memory.total --format=csv -l 60 > "$JOBDIR/gpu.csv" &
SMI=$!

# --- the run ------------------------------------------------------------------
# stdout: spin / mask-rejection chatter goes to its own files; tracebacks go to the .err file
python -u scripts/tmnre_joint.py \
    --train-config "$TRAIN_CONFIG" \
    --datagen-config "$DATAGEN_CONFIG" \
    --obs-path "$OBS" \
    "${ARGS[@]}" \
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
    grep -h "Stopped after epoch" "$REPO/slurm/logs/${SLURM_JOB_NAME}_${SLURM_JOB_ID}.out" | tail -3
    if [[ $status -ne 0 ]]; then
        echo "--- last 30 lines of stderr ---"
        tail -n 30 "$REPO/slurm/logs/${SLURM_JOB_NAME}_${SLURM_JOB_ID}.err"
    fi
} | tee "$JOBDIR/summary.txt"
exit $status
