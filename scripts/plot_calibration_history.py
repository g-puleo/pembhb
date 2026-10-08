#!/usr/bin/env python
"""Calibration of the 1D posteriors on the test pool, across training.

``CalibrationMonitor`` evaluates, every epoch, the posterior rank of the true
value for each of the first ``test_n`` test-pool simulations, and logs per 1D
marginal (TensorBoard step = cumulative epoch over all rounds):

* ``D`` — KS distance of the ranks from Uniform(0, 1); 0 when calibrated;
* ``T`` — fraction of ranks in the outer ``2q`` tails; ``2q`` when calibrated,
  above it when overconfident, below it when underconfident.

One row per marginal: D (raw + EMA) and T (raw + EMA), with the reference
levels ``d_threshold`` and ``2q ± t_threshold`` from the run's config and
vertical lines at round boundaries.

    python scripts/plot_calibration_history.py RUN_TAG

Writes ``$PEMBHB_PLOTS_DIR/<run>/calibration_history.png``.
"""

import argparse
import os
from glob import glob

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml
from tensorboard.backend.event_processing import event_accumulator

from pembhb import DATA_ROOT_DIR, PLOTS_ROOT_DIR


def _last_version_dir(run_name: str, round_idx: int) -> str | None:
    vers = sorted(glob(os.path.join(DATA_ROOT_DIR, "logs", run_name,
                                    f"round_{round_idx}", "version_*")))
    return vers[-1] if vers else None


def _accumulator(version_dir: str) -> event_accumulator.EventAccumulator:
    ea = event_accumulator.EventAccumulator(
        version_dir, size_guidance={event_accumulator.SCALARS: 0})
    ea.Reload()
    return ea


def _scalars(ea, tag: str):
    if tag not in ea.Tags()["scalars"]:
        return np.array([]), np.array([])
    events = ea.Scalars(tag)
    return np.array([e.step for e in events]), np.array([e.value for e in events])


def _labels(ea) -> list[str]:
    return sorted(t.split("/", 2)[2] for t in ea.Tags()["scalars"]
                  if t.startswith("pp_ks/D/"))


def load_history(run_name: str):
    """``({label: {"epoch", "D", "T", "D_ema", "T_ema"}}, round_ends)``."""
    accs = []
    r = 1
    while (vdir := _last_version_dir(run_name, r)) is not None:
        accs.append(_accumulator(vdir))
        r += 1
    if not accs:
        raise SystemExit(f"No rounds found for run {run_name} under {DATA_ROOT_DIR}/logs.")
    labels = sorted({lbl for ea in accs for lbl in _labels(ea)})
    if not labels:
        raise SystemExit("No pp_ks/D/* scalars found: was calibration_monitor enabled?")

    out = {lbl: {k: [] for k in ("epoch", "D", "T", "D_ema", "T_ema")} for lbl in labels}
    round_ends = []
    for ea in accs:
        last = None
        for lbl in labels:
            steps, d = _scalars(ea, f"pp_ks/D/{lbl}")
            if steps.size == 0:
                continue
            _, t = _scalars(ea, f"pp_ks/T/{lbl}")
            # the EMAs start after warmup: align them to the raw steps
            for key in ("D_ema", "T_ema"):
                s_e, v_e = _scalars(ea, f"pp_ks/{key}/{lbl}")
                lut = dict(zip(s_e.tolist(), v_e.tolist()))
                out[lbl][key].extend(lut.get(s, np.nan) for s in steps.tolist())
            out[lbl]["epoch"].extend(steps.tolist())
            out[lbl]["D"].extend(d.tolist())
            out[lbl]["T"].extend(t.tolist())
            last = max(last or 0, int(steps.max()))
        if last is not None:
            round_ends.append(last + 0.5)
    for lbl in labels:
        order = np.argsort(out[lbl]["epoch"], kind="stable")
        for k in out[lbl]:
            out[lbl][k] = np.asarray(out[lbl][k], dtype=float)[order]
    return out, round_ends


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_tag", help="run tag (TIME_OF_EXECUTION)")
    args = ap.parse_args()

    with open(os.path.join(DATA_ROOT_DIR, args.run_tag, "calibration_state.yaml")) as f:
        state = yaml.safe_load(f)
    d_thr, t_thr, q = state["d_threshold"], state["t_threshold"], state["t_quantile"]
    t_base = 2.0 * q

    hist, round_ends = load_history(args.run_tag)
    labels = list(hist)
    fig, axes = plt.subplots(len(labels), 2, figsize=(11, 1.9 * len(labels)),
                             sharex=True, squeeze=False)
    for row, lbl in enumerate(labels):
        h = hist[lbl]
        ax_d, ax_t = axes[row]
        first = row == 0
        ax_d.plot(h["epoch"], h["D"], lw=0.6, alpha=0.5, color="tab:blue",
                  label="raw" if first else None)
        ax_d.plot(h["epoch"], h["D_ema"], lw=1.4, color="tab:blue",
                  label="EMA" if first else None)
        ax_t.plot(h["epoch"], h["T"], lw=0.6, alpha=0.5, color="tab:orange",
                  label="raw" if first else None)
        ax_t.plot(h["epoch"], h["T_ema"], lw=1.4, color="tab:orange",
                  label="EMA" if first else None)
        ax_d.axhline(d_thr, color="red", ls=":", lw=0.8,
                     label=f"d_threshold={d_thr}" if first else None)
        ax_t.axhline(t_base, color="grey", lw=0.6,
                     label=f"2q={t_base:.2f} (calibrated)" if first else None)
        for sgn in (1, -1):
            ax_t.axhline(t_base + sgn * t_thr, color="red", ls=":", lw=0.8,
                         label=f"2q ± t_threshold" if first and sgn > 0 else None)
        for b in round_ends[:-1]:
            ax_d.axvline(b, color="k", lw=0.4, alpha=0.3)
            ax_t.axvline(b, color="k", lw=0.4, alpha=0.3)
        ax_d.set_ylabel(f"{lbl}\nD", fontsize=9)
        ax_t.set_ylabel("T", fontsize=9)
        ax_d.set_ylim(0, 1)
        t_max = np.nanmax(h["T"]) if np.isfinite(h["T"]).any() else 0.3
        ax_t.set_ylim(0, max(0.3, 1.1 * t_max))
        if first:
            ax_d.legend(fontsize=7, loc="upper right")
            ax_t.legend(fontsize=7, loc="upper right")
    for ax in axes[-1]:
        ax.set_xlabel("cumulative training epoch")
    fig.suptitle(f"calibration on the test pool — {args.run_tag}", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.98))

    out_dir = os.path.join(PLOTS_ROOT_DIR, args.run_tag)
    os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, "calibration_history.png")
    fig.savefig(out, dpi=120)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
