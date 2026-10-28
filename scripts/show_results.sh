#!/usr/bin/env bash
# Make every results figure of a TMNRE run (README §5-§6) in one go.
#
#   scripts/show_results.sh RUN_TAG [--round N] [--mcmc-file FILE] [--obs-path FILE] [--no-proposal] [--fisher]
#
#   RUN_TAG        run tag printed by tmnre_joint.py, e.g. 2026/10/08/channelizedmlp_my_run
#   --round N      posterior of round N and evolution up to round N (default: all rounds)
#   --mcmc-file    flat MCMC samples HDF5, overlaid where supported
#   --obs-path     local copy of the observation (default: observation_used.yaml);
#                  checked against the recorded obs_hash
#   --no-proposal  do not overlay the next-round truncation on the posterior
#   --quick        only the posterior of the last round (or --round N), with the
#                  MCMC overlay if --mcmc-file is given; skips all evolution plots
#   --fisher       overlay the Fisher (Cramer-Rao) Gaussian on the 1D evolution
#
# Needs PEMBHB_DATA_DIR / PEMBHB_PLOTS_DIR set as for training, and `python`
# from the pembhb environment on PATH. A failing step is reported and skipped.

set -uo pipefail

usage() { sed -n '2,17p' "$0" | sed 's/^# \{0,1\}//'; exit "${1:-0}"; }

[[ $# -ge 1 ]] || usage 1
[[ $1 == -h || $1 == --help ]] && usage 0
TAG=$1; shift
ROUND=""; MCMC=""; OBS=""; PROPOSAL="--proposal"; QUICK=""; FISHER=()
while [[ $# -gt 0 ]]; do
    case $1 in
        --round)       ROUND=$2; shift 2 ;;
        --mcmc-file)   MCMC=$2; shift 2 ;;
        --obs-path)    OBS=$2; shift 2 ;;
        --no-proposal) PROPOSAL=""; shift ;;
        --quick)       QUICK=1; shift ;;
        --fisher)      FISHER=(--fisher); shift ;;
        -h|--help)     usage 0 ;;
        *) echo "unknown option: $1" >&2; usage 1 ;;
    esac
done

HERE=$(cd "$(dirname "$0")" && pwd)
DATA=${PEMBHB_DATA_DIR:-$HERE/../data}
PLOTS=${PEMBHB_PLOTS_DIR:-$HERE/../plots}
[[ -d $DATA/logs/$TAG ]] || { echo "no run '$TAG' under $DATA/logs (is PEMBHB_DATA_DIR set?)" >&2; exit 1; }

mcmc=(); [[ -n $MCMC ]] && mcmc=(--mcmc-file "$MCMC")
obs=(); dpath=(); [[ -n $OBS ]] && { obs=(--obs-path "$OBS"); dpath=(--data-path "$OBS"); }
upto=(); [[ -n $ROUND ]] && upto=(--last-round "$ROUND")
round=(); [[ -n $ROUND ]] && round=(--round "$ROUND")

failed=()
step() {
    local name=$1; shift
    echo; echo "=== $name"
    "$@" || { echo "!!! $name failed (exit $?)"; failed+=("$name"); }
}

step "posterior of one round"   python "$HERE/plot_posterior.py" --run-name "$TAG" "${round[@]}" "${mcmc[@]}" "${obs[@]}" $PROPOSAL
if [[ -n $QUICK ]]; then
    echo; echo "Figures: $PLOTS/$TAG/posterior_round_*/"
    [[ ${#failed[@]} -eq 0 ]] || exit 1
    exit 0
fi
step "1D evolution"             python "$HERE/visualise_truncation_rounds.py" "$TAG" "${upto[@]}" "${mcmc[@]}" "${dpath[@]}" "${FISHER[@]}"
step "sky evolution"            python "$HERE/visualise_sky_truncation.py" "$TAG" "${upto[@]}" "${mcmc[@]}" "${dpath[@]}"
step "entropy evolution"        python "$HERE/visualise_entropy_evolution.py" "$TAG" "${upto[@]}" "${dpath[@]}"
step "prior volume"             python "$HERE/visualise_volume_from_masks.py" "$TAG" "${upto[@]}"
if [[ -f $DATA/$TAG/calibration_state.yaml || -f $DATA/$TAG/ppks_state.yaml ]]; then
    step "calibration history"  python "$HERE/plot_calibration_history.py" "$TAG"
else
    echo; echo "=== calibration history: skipped (calibration_monitor was off)"
fi
if compgen -G "$DATA/$TAG/lambda_tau_stats_round_*.h5" > /dev/null; then
    step "lambda-tau plane"     python "$HERE/plot_lambda_tau.py" "$TAG"
else
    echo; echo "=== lambda-tau plane: skipped (calibration_monitor.lambda_tau was off)"
fi

suffix=""; [[ -n $ROUND ]] && suffix="_upto_round_$ROUND"
echo
echo "Figures:"
echo "  $PLOTS/$TAG/posterior_round_*/      one round's posterior"
echo "  $PLOTS/$TAG$suffix/                 evolution across rounds (+ sky/)"
echo "  $PLOTS/$TAG/                        calibration_history.png, lambda_tau_plane.png"
echo "Training curves:"
echo "  tensorboard --logdir $DATA/logs/$TAG"
if [[ ${#failed[@]} -gt 0 ]]; then
    echo; echo "Failed steps: ${failed[*]}"; exit 1
fi
