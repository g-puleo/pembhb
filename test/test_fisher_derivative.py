"""Convergence DIAGNOSTIC for the Fisher-matrix waveform derivative.

This is a diagnostic, not a pass/fail test: it produces per-parameter log-log
plots, recommends a good finite-difference step `dx`, and saves the Richardson
"ground truth" derivative evaluated at that recommended step (the derivative you
should actually feed into the Fisher matrix).

Theory
------
The Fisher matrix F_ij = <d_i h | d_j h> is built from a central-difference
derivative of the waveform vector h(x) (complex, over channel x frequency bins):

    f'_approx(x, dx) = ( h(x + dx) - h(x - dx) ) / (2 dx) .

Taylor-expanding componentwise to 3rd order:

    f'_approx(x, dx) = h'(x) + C dx^2 + O(dx^4) ,     C = h'''(x) / 6 ,

so the Richardson extrapolation that cancels the dx^2 term is

    f'_gt(x, dx) = (1/3)( 4 f'_approx(x, dx) - f'_approx(x, 2 dx) ) = h'(x) + O(dx^4) .

The diagnostic measures

    e(dx) = || f'_approx(dx) - f'_gt(dx) || = (1/3) || f'_approx(2dx) - f'_approx(dx) ||
          = || C || dx^2 + O(dx^4) ,

so log||e(dx)|| = log||C|| + n log(dx) with n = 2 (dx is the same for every bin,
so it factors out of the norm; the per-bin constants only set the intercept).

Why the elbow / what "too small" means
---------------------------------------
Floating point adds a competing roundoff error from catastrophic cancellation in
h(x+dx) - h(x-dx) (two nearly-equal vectors): || e_round(dx) || ~ eps_mach ||h|| / dx,
i.e. slope -1. So total error ~ ||C|| dx^2 + (eps_mach ||h|| / 2) dx^-1 has a
minimum (the elbow) at dx* ~ eps_mach^(1/3): the optimal step. Left of the elbow
the central difference is roundoff-limited (corrupted); right of it, truncation-
limited. A production step that sits left of the elbow is "too small".

Recommended step
----------------
We recommend the elbow dx (the directly-measured error minimum) and evaluate the
Richardson ground truth f_gt there. f_gt cancels the dx^2 truncation, so at the
elbow it is roundoff-limited and about as accurate as the data allows.

Outputs (written to test/output/)
---------------------------------
  * fisher_derivative_convergence.png  -- per-parameter log-log plots
  * fisher_derivative_recommended.npz  -- recommended dx + f_gt derivative per param
  * fisher_derivative_summary.txt      -- table: production dx, recommended dx, verdict

Run:
    python -m pytest test/test_fisher_derivative.py -s
or standalone:
    python test/test_fisher_derivative.py
"""

import os
import copy
import warnings

import numpy as np
import pytest
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pembhb import ROOT_DIR
from pembhb.utils import (
    read_config,
    waveform_central_difference,
    FISHER_ABSOLUTE_STEP_DEFAULTS,
    _ORDERED_PRIOR_KEYS,
)
from pembhb.simulator import MBHBSimulatorFD

# Powers-of-two ladder of multipliers on each parameter's production step. Using
# exact factors of 2 means the Richardson pair (dx, 2dx) are always *adjacent*
# ladder points, so each central difference is computed only once. Range 2^5 ..
# 2^-6 (32x down to ~0.016x) brackets the elbow for every parameter.
LOG2_FACTORS = np.arange(5, -7, -1)          # [5, 4, ..., -6]  -> descending dx
DX_FACTORS = 2.0 ** LOG2_FACTORS             # [32, 16, ..., 0.0156]

OUTPUT_DIR = os.path.join(ROOT_DIR, "test", "output")


# ---------------------------------------------------------------------------
# Simulator / expansion point
# ---------------------------------------------------------------------------
def _build_simulator(config_path=os.path.join(ROOT_DIR, "test", "configs", "datagen_test.yaml")):
    """CPU simulator from a datagen config (deterministic, GPU-free).

    Use a config whose frequency grid / noise model matches the observation you
    plan to expand at (the ``--obs`` grid-consistency assert will catch a
    mismatch).
    """
    conf = copy.deepcopy(read_config(config_path))
    conf["backend"] = "cpu"
    wp = conf["waveform_params"]
    return MBHBSimulatorFD(
        conf,
        sampler_init_kwargs={"prior_bounds": conf["prior"],
                             "spin_param_basis": conf.get("spin_param_basis", "chi1chi2")},
        seed=42,
        n_freq_bins=wp.get("n_freq_bins", 4096),
        freq_spacing=wp.get("freq_spacing", "linear"),
    )


def _draw_theta0(simulator):
    """One prior sample as the expansion point, in TMNRE (_ORDERED_PRIOR_KEYS) coords."""
    _, tmnre_input = simulator.sampler.sample(1, t_obs_end=simulator.t_obs_end_SI)
    return np.asarray(tmnre_input[:, 0], dtype=np.float64)


def _theta0_from_obs(simulator, obs_path, event_idx=0):
    """Expansion point read from an observation file (TMNRE coords).

    Uses the true ``source_parameters[event_idx]`` of a real observation rather
    than a prior draw, and asserts the simulator's frequency grid matches the
    one the observation was generated on (else the derivative would be taken on
    an inconsistent grid).
    """
    import h5py
    with h5py.File(obs_path, "r") as f:
        freqs_obs = f["frequencies"][:]
        theta0 = np.asarray(f["source_parameters"][event_idx], dtype=np.float64)
    assert freqs_obs.shape == simulator.freqs.shape and np.allclose(freqs_obs, simulator.freqs), (
        f"frequency-grid mismatch between {obs_path} and the simulator built from "
        f"the datagen config; rebuild the simulator with a matching config."
    )
    return theta0


@pytest.fixture(scope="module")
def simulator_and_theta0():
    sim = _build_simulator()
    return sim, _draw_theta0(sim)


# ---------------------------------------------------------------------------
# Core convergence computation
# ---------------------------------------------------------------------------
def _convergence_for_param(simulator, theta0, param_name):
    """Sweep the dx ladder for one parameter and recommend a step.

    Returns a dict with:
      dx_err       : dx values at which the error is defined (descending)
      errors       : || f'_approx(2dx) - f'_approx(dx) || / 3  at each dx_err
      slope, icpt  : log-log fit over the truncation band (largest dx -> elbow)
      fit_mask     : which dx_err points were used in the fit
      prod_dx      : the production step (FISHER_ABSOLUTE_STEP_DEFAULTS)
      rec_dx       : recommended step = elbow (error minimum)
      f_gt         : Richardson ground-truth derivative (n_ch, n_freq) at rec_dx
      verdict      : 'ok', 'too_small', 'elbow_below_range', or 'elbow_above_range'
    """
    prod_dx = FISHER_ABSOLUTE_STEP_DEFAULTS[param_name]
    ladder = DX_FACTORS * prod_dx            # descending; ladder[i-1] == 2*ladder[i]

    # One central difference per distinct ladder step.
    deriv = [
        waveform_central_difference(simulator, theta0, param_name, float(dx))
        for dx in ladder
    ]

    # Error is defined for every ladder point that has a 2x partner above it,
    # i.e. indices 1..end. e(ladder[i]) uses f'_approx(ladder[i]) and its 2x
    # partner f'_approx(ladder[i-1]).
    dx_err = ladder[1:]
    errors = np.array([
        np.linalg.norm((deriv[i - 1] - deriv[i]).ravel()) / 3.0
        for i in range(1, len(ladder))
    ])

    # Elbow = error minimum. We do NOT recommend the elbow itself: there
    # truncation ~ roundoff, so it sits off the clean dx^2 line. Instead pick the
    # point one ladder step toward LARGER dx (index j-1, since dx_err descends),
    # which is firmly in the dx^2 truncation regime -> a reliable step at which
    # to evaluate the Richardson ground truth.
    j = int(np.argmin(errors))               # elbow index into dx_err / errors
    elbow_dx = float(dx_err[j])
    rec_m = max(j - 1, 0)                     # one step into the dx^2 region
    rec_dx = float(dx_err[rec_m])

    # Richardson ground truth at rec_dx == ladder[rec_m+1]; its 2x partner is
    # ladder[rec_m] -> reuse the cached central differences.
    f_gt = (4.0 * deriv[rec_m + 1] - deriv[rec_m]) / 3.0

    # Fit the truncation band: points STRICTLY to the right of the elbow (larger
    # dx), i.e. the pure +2 branch. The elbow itself is excluded because there
    # truncation ~ roundoff (it sits ~2x above the pure-truncation value and
    # biases the slope low). If excluding it leaves < 2 points (elbow near the
    # largest dx -> range set too high), widen to the first 3 points.
    fit_mask = np.zeros_like(errors, dtype=bool)
    fit_mask[:j] = True
    if fit_mask.sum() < 2:
        fit_mask[: min(3, len(errors))] = True
    slope, icpt = np.polyfit(np.log10(dx_err[fit_mask]), np.log10(errors[fit_mask]), 1)

    # Verdict relative to the production step. "ok" means the production step is
    # at or above the recommended dx^2-region point; "too_small" means it sits
    # below it (toward the elbow / roundoff branch).
    if j == 0:
        verdict = "elbow_above_range"        # optimum is even larger than swept
    elif j == len(errors) - 1:
        verdict = "elbow_below_range"        # optimum below swept range (step has margin)
    elif prod_dx < rec_dx * (1.0 - 1e-9):
        verdict = "too_small"                # production below the clean dx^2 region
    else:
        verdict = "ok"

    return dict(dx_err=dx_err, errors=errors, slope=slope, icpt=icpt,
                fit_mask=fit_mask, prod_dx=prod_dx, rec_dx=rec_dx, elbow_dx=elbow_dx,
                f_gt=f_gt, verdict=verdict)


# ---------------------------------------------------------------------------
# Plot + summary writers
# ---------------------------------------------------------------------------
def make_plot(results, out_path):
    """Grid of per-parameter log-log convergence plots."""
    names = list(results)
    n = len(names)
    ncols = 4
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3.2 * nrows))
    axes = np.atleast_1d(axes).ravel()

    for ax, name in zip(axes, names):
        r = results[name]
        ax.loglog(r["dx_err"], r["errors"], "o", color="C0", label="||error(dx)||")
        dxf = r["dx_err"][r["fit_mask"]]
        ax.loglog(dxf, 10 ** r["icpt"] * dxf ** r["slope"], "-", color="C3",
                  label=f"fit n={r['slope']:.2f}")
        ax.axvline(r["prod_dx"], color="k", ls="--", lw=0.8,
                   label=f"prod={r['prod_dx']:.0e}")
        ax.axvline(r["rec_dx"], color="C2", ls="-", lw=1.0,
                   label=f"rec={r['rec_dx']:.1e}")
        # Elbow marker (error minimum) — recommendation is one step to its right.
        je = int(np.argmin(r["errors"]))
        ax.plot(r["dx_err"][je], r["errors"][je], "s", color="C1", ms=6,
                label="elbow")
        ax.set_title(f"{name}  [{r['verdict']}]",
                     color=("C3" if r["verdict"] == "too_small" else "k"))
        ax.set_xlabel("dx")
        ax.set_ylabel("||f'_approx - f'_gt||")
        ax.legend(fontsize=6, loc="best")
        ax.grid(True, which="both", alpha=0.3)

    for ax in axes[n:]:
        ax.set_visible(False)

    fig.suptitle(
        "Fisher waveform-derivative convergence (slope n->2 = truncation branch; "
        "elbow = recommended dx; prod left of elbow => too small)", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=130)
    print(f"[fisher-deriv] saved convergence plot -> {out_path}")


def write_outputs(results, tag=""):
    """Save the recommended dx + f_gt derivatives and a human-readable summary."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    npz_path = os.path.join(OUTPUT_DIR, f"fisher_derivative_recommended{tag}.npz")
    txt_path = os.path.join(OUTPUT_DIR, f"fisher_derivative_summary{tag}.txt")

    npz = {}
    for name, r in results.items():
        npz[f"{name}__rec_dx"] = np.array(r["rec_dx"])
        npz[f"{name}__f_gt"] = r["f_gt"]
    np.savez(npz_path, **npz)

    lines = [
        f"{'param':>10} {'prod_dx':>10} {'elbow_dx':>10} {'rec_dx':>10} "
        f"{'rec/prod':>9} {'slope_n':>8} {'||f_gt||':>12}  verdict",
    ]
    for name, r in results.items():
        lines.append(
            f"{name:>10} {r['prod_dx']:>10.2e} {r['elbow_dx']:>10.2e} "
            f"{r['rec_dx']:>10.2e} {r['rec_dx'] / r['prod_dx']:>9.2f} "
            f"{r['slope']:>8.3f} {np.linalg.norm(r['f_gt'].ravel()):>12.4e}  "
            f"{r['verdict']}"
        )
    summary = "\n".join(lines)
    with open(txt_path, "w") as f:
        f.write(summary + "\n")
    print(f"[fisher-deriv] recommended steps & f_gt -> {npz_path}")
    print(summary)
    return summary


def _run_all(simulator, theta0):
    results = {}
    for name in _ORDERED_PRIOR_KEYS:
        results[name] = _convergence_for_param(simulator, theta0, name)
    return results


# ---------------------------------------------------------------------------
# pytest entry point (diagnostic: always runs to completion, only warns)
# ---------------------------------------------------------------------------
def test_fisher_derivative_diagnostic(simulator_and_theta0):
    """Produce the convergence plots + recommended steps; warn on too-small steps."""
    simulator, theta0 = simulator_and_theta0
    results = _run_all(simulator, theta0)

    make_plot(results, os.path.join(OUTPUT_DIR, "fisher_derivative_convergence.png"))
    write_outputs(results)

    for name, r in results.items():
        if r["verdict"] == "too_small":
            warnings.warn(
                f"[{name}] production step {r['prod_dx']:.2e} is below the clean "
                f"dx^2 region (elbow {r['elbow_dx']:.2e}, recommended "
                f"{r['rec_dx']:.2e} = {r['rec_dx'] / r['prod_dx']:.1f}x larger); "
                f"the derivative is roundoff-affected. Bump the step toward rec_dx."
            )

    # Sanity only: the errors must be finite (catches NaN waveform blow-ups).
    for name, r in results.items():
        assert np.all(np.isfinite(r["errors"])), f"[{name}] non-finite errors"


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--obs", default=None,
                    help="Observation HDF5 file: use its source_parameters[event] "
                         "as the expansion point (instead of a prior draw).")
    ap.add_argument("--event", type=int, default=0,
                    help="Event index within --obs (default 0).")
    ap.add_argument("--config", default=os.path.join(ROOT_DIR, "configs", "datagen_config.yaml"),
                    help="Datagen config (grid/noise must match --obs).")
    ap.add_argument("--tag", default=None,
                    help="Override output-file suffix (default: '_obs' with --obs, else '').")
    args = ap.parse_args()

    sim = _build_simulator(args.config)
    if args.obs:
        th0 = _theta0_from_obs(sim, args.obs, args.event)
        tag = "_obs" if args.tag is None else args.tag
        print(f"[fisher-deriv] expansion point: {args.obs} event {args.event}")
    else:
        th0 = _draw_theta0(sim)
        tag = "" if args.tag is None else args.tag
        print("[fisher-deriv] expansion point: prior draw from datagen_config (seed 42)")

    res = _run_all(sim, th0)
    make_plot(res, os.path.join(OUTPUT_DIR, f"fisher_derivative_convergence{tag}.png"))
    write_outputs(res, tag=tag)
