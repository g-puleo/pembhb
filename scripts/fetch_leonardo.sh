#!/usr/bin/env bash
# Pull a Leonardo run to this machine and make its figures here.
#
#   scripts/fetch_leonardo.sh RUN_TAG [--no-plots] [show_results.sh options]
#
#   RUN_TAG      e.g. 2026/10/27/channelizedmlp_het84d_fisher10
#   --no-plots   only fetch, do not run show_results.sh
#   other options are passed on to show_results.sh (--round, --mcmc-file, --fisher, ...)
#
# rsync copies only what changed since the last fetch:
#   data/RUN_TAG/   yamls, masks, stats (not simulation_round_*.h5, stream_buffers/)
#   logs/RUN_TAG/   truncation.ckpt, tfevents, yamls; checkpoints/ only for a round
#                   still training (no truncation.ckpt yet)
#   plots/RUN_TAG/  posterior_evolution/, ppks_traces/ (written during training)
# If --obs-path is not given, the observation is looked up by file name under
# PEMBHB_DATA_DIR; show_results.sh checks it against the recorded obs_hash.
#
# Needs a valid cert (step ssh login '<email>' --provisioner cineca-hpc) and
# PEMBHB_DATA_DIR / PEMBHB_PLOTS_DIR set. LEONARDO_HOST (default: leonardo) and
# LEONARDO_ROOT (default: /leonardo_work/Sis26_trotta/gpuleo00/pembhb) set the remote side.

set -uo pipefail

usage() { sed -n '2,21p' "$0" | sed 's/^# \{0,1\}//'; exit "${1:-0}"; }

[[ $# -ge 1 ]] || usage 1
[[ $1 == -h || $1 == --help ]] && usage 0
TAG=${1%/}; shift
PLOT=1; HAS_OBS=""; PASS=()
while [[ $# -gt 0 ]]; do
    case $1 in
        --no-plots) PLOT=""; shift ;;
        --obs-path) HAS_OBS=1; PASS+=("$1" "$2"); shift 2 ;;
        *)          PASS+=("$1"); shift ;;
    esac
done

: "${PEMBHB_DATA_DIR:?set PEMBHB_DATA_DIR}" "${PEMBHB_PLOTS_DIR:?set PEMBHB_PLOTS_DIR}"
HERE=$(cd "$(dirname "$0")" && pwd)
DATA=$PEMBHB_DATA_DIR
PLOTS=$PEMBHB_PLOTS_DIR
HOST=${LEONARDO_HOST:-leonardo}
R=${LEONARDO_ROOT:-/leonardo_work/Sis26_trotta/gpuleo00/pembhb}

ssh -o BatchMode=yes -o ConnectTimeout=10 "$HOST" true || {
    echo "cannot ssh to $HOST: run  step ssh login '<email>' --provisioner cineca-hpc" >&2; exit 1; }
ssh "$HOST" test -d "$R/data/logs/$TAG" || { echo "no run '$TAG' under $HOST:$R/data/logs" >&2; exit 1; }

mkdir -p "$DATA/$TAG" "$DATA/logs/$TAG" "$PLOTS/$TAG"

echo "=== data/$TAG"
rsync -a --exclude='stream_buffers/' --exclude='simulation_round_*.h5' --exclude='*.tmp' \
    "$HOST:$R/data/$TAG/" "$DATA/$TAG/" || exit 1

echo "=== logs/$TAG"
rsync -a --exclude='checkpoints/' "$HOST:$R/data/logs/$TAG/" "$DATA/logs/$TAG/" || exit 1
for r in "$DATA/logs/$TAG"/round_*/; do
    v=$(ls -d "$r"version_* 2>/dev/null | sort -V | tail -1)
    [[ -n $v && ! -f $v/truncation.ckpt ]] || continue
    rel=${v#"$DATA"/}
    echo "    $(basename "$r") still training: fetching its checkpoints/"
    rsync -a "$HOST:$R/data/$rel/checkpoints/" "$v/checkpoints/" || echo "    (no checkpoint yet)"
done

echo "=== plots/$TAG"
rsync -a --ignore-missing-args "$HOST:$R/plots/$TAG/posterior_evolution" "$HOST:$R/plots/$TAG/ppks_traces" \
    "$PLOTS/$TAG/" || echo "    (no training-time plots yet)"

[[ -n $PLOT ]] || { echo; echo "Fetched to $DATA/$TAG, $DATA/logs/$TAG, $PLOTS/$TAG"; exit 0; }

if [[ -z $HAS_OBS ]]; then
    used=$(sed -n 's/^obs_path: *//p' "$DATA/$TAG/observation_used.yaml" 2>/dev/null)
    if [[ -n $used && ! -f $used ]]; then
        obs=$(find "$DATA" -maxdepth 4 \( -path "$DATA/20*" -o -path "$DATA/logs" -o -path "$DATA/mcmc" \) -prune \
                   -o -name "$(basename "$used")" -print -quit)
        [[ -n $obs ]] || { echo "no local copy of $(basename "$used") under $DATA: pass --obs-path" >&2; exit 1; }
        echo "=== observation: $obs"
        PASS+=(--obs-path "$obs")
    fi
fi

exec "$HERE/show_results.sh" "$TAG" "${PASS[@]}"
