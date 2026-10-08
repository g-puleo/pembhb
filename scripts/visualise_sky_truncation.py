"""Sky-localisation visualisation across TMNRE truncation rounds.

Resurrects the sky-marginal figures that lived in the pre-refactor
``visualise_truncation_rounds.py``, restructured as a standalone script and
built on the shared helpers in ``_visualise_common`` / ``viz_helpers``.

Produces six figures under ``PLOTS_ROOT_DIR/{name}/sky/``:

1. ``final_posterior_mcmc.pdf`` — the last round's NRE sky posterior overlaid
   with the MCMC posterior for the same observation (Mollweide, 50% & 90% CR).
2. ``sky_area_vs_round.pdf`` — 90% credible-region sky area (sq deg) per round,
   with the MCMC posterior area as a horizontal baseline.
3. ``truncation_scheme_mollweide.pdf`` — one panel per ``--rounds`` round in
   Mollweide coordinates, overlaying that round's *training-prior mask* (the
   truncation from the previous round), its NRE sky posterior, and the
   *proposed mask* for the next round.  In mask-truncation runs these regions
   are the actual HPD masks (read from ``truncation_round_{i}.npz``), not boxes;
   in Mollweide their constant-lambda edges bow as meridians, because
   latitude = arcsin(sin beta).
4. ``truncation_scheme_native.pdf`` — the same per-round overlays in native
   (lambda, sin beta) coordinates, zoomed to the training-prior mask.
5. ``final_posterior_zoom.pdf`` — the final round's NRE posterior (+ MCMC),
   re-evaluated on a fine grid over the credible-region bounding box, in native
   (lambda [rad], sin beta) coordinates with a dashed 99% contour.
6. ``final_posterior_zoom_lonlat.pdf`` — the same zoom, with the axes remapped
   to ecliptic longitude / latitude in degrees.

Parametrisation: slot 7 is lambda (ecliptic longitude, [0, 2pi]); slot 8 is
sin(ecliptic latitude), NOT the latitude itself.  The sky solid-angle element
is dOmega = dlambda * d(sin beta), so sky area is a native-grid cell count.

Usage
-----
    python scripts/visualise_sky_truncation.py NAME \\
        --data-path obs_withnoise.h5 [--mcmc-file flat_samples.h5] \\
        [--rounds R [R ...]] [--ngrid 200] [--last-round N] \\
        [--ckpt-final-round PATH] [--cr-area 0.90]
"""

import argparse
import math
import os

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from scipy.stats import gaussian_kde

from pembhb import ROOT_DIR, PLOTS_ROOT_DIR
from pembhb.utils import eval_posterior_2d, contour_levels
from pembhb.mask_truncation import load_pair_region, rasterise_region

from _visualise_common import (
    DATA_ROOT_DIR,
    resolve_obs_path,
    build_obs_dataloader,
    find_round_dirs,
    load_model,
    load_prior_box,
    find_out_param_idx,
    keys_for_model,
    register_ckpt_override,
    PT_PER_INCH,
)
from viz_helpers import (
    load_mcmc_samples,
    to_mollweide_coords,
    compute_sky_area,
    is_full_sky,
)

# Canonical (lambda, beta) parameter indices.
SKY = (7, 8)
_CR_CONTOURS = (0.50, 0.90)
# Zoom figure draws 50/90% solid + 99% dashed.
_CR_ZOOM = (0.50, 0.90, 0.99)
_CR_ZOOM_DASHED = 0.99

# final_posterior_zoom.pdf is placed in presentations/poster_eucaif/poster_eucaif.tex
# at native size (no \includegraphics width= rescaling). Sized to 90% of the
# figure's previous DISPLAYED height in that slot (7.0917in wide * 0.84205
# native aspect = 5.9715in displayed height; 90% = 5.3743in), width scaled to
# preserve the original figsize's 7:6 aspect ratio. Fontsize matched to the
# poster's body text (29pt, see plot_cherry_posterior_1d.py) with matplotlib's
# "cm" mathtext (not text.usetex) to match the poster's own LaTeX serif font.
_ZOOM_FIG_WIDTH_IN = 6.270
_ZOOM_FIG_HEIGHT_IN = 5.374
_ZOOM_FONTSIZE = 29

# final_posterior_mcmc.pdf legacy default (no paper styling): a bare 10x6in
# canvas at matplotlib's ambient rcParams. --width-pt/--fontsize (see main())
# switch it to a paper single-column figure instead -- serif/cm mathtext,
# uniform fontsize, thinner lines, no title (caption's job), sparser graticule.
_FINAL_FIG_WIDTH_IN = 10.0
_FINAL_FIG_HEIGHT_IN = 6.0


# ---------------------------------------------------------------------------
# Sky-box geometry
# ---------------------------------------------------------------------------

def _sky_box_loop(lam_lo, lam_hi, sinb_lo, sinb_hi, mollweide, n=200):
    """Return (x, y) of the closed box boundary in the chosen coordinates.

    The boundary is densely sampled along the four edges *in native
    (lambda, sin beta)* and then mapped.  In Mollweide this yields the correct
    curvilinear patch (constant-lambda edges become curved meridians); in
    native coordinates it traces an axis-aligned rectangle.
    """
    edge = np.linspace
    lam = np.concatenate([
        edge(lam_lo, lam_hi, n),          # bottom (sin b = sinb_lo)
        np.full(n, lam_hi),               # right  (lambda = lam_hi)
        edge(lam_hi, lam_lo, n),          # top    (sin b = sinb_hi)
        np.full(n, lam_lo),               # left   (lambda = lam_lo)
    ])
    sinb = np.concatenate([
        np.full(n, sinb_lo),
        edge(sinb_lo, sinb_hi, n),
        np.full(n, sinb_hi),
        edge(sinb_hi, sinb_lo, n),
    ])
    if mollweide:
        return lam - np.pi, np.arcsin(np.clip(sinb, -1.0, 1.0))
    return lam, sinb


def _load_sky_mask(name, round_number):
    """Boolean sky mask saved at the END of ``round_number`` — i.e. the proposal
    that became the prior for round ``round_number + 1`` — in native
    (lambda, sin beta).

    Returns ``(mask, lam_axis, sinb_axis)`` with ``mask`` shape
    ``(n_sinb, n_lam)`` (True inside the region), or ``None`` when no mask file
    or sky pair exists (round 0, or a rectangle-mode run).
    """
    if round_number < 1:
        return None
    npz = os.path.join(DATA_ROOT_DIR, name, f"truncation_round_{round_number}.npz")
    if not os.path.exists(npz):
        return None
    # Both npz formats go through load_pair_region; format 2 keeps each mode on
    # its own subgrid, so rasterise onto one common grid for plotting.
    i, j = SKY
    region = load_pair_region(npz, i, j)
    if region is not None:                             # stored (lambda, sin beta)
        return rasterise_region(region)
    region = load_pair_region(npz, j, i)
    if region is not None:                             # stored (sin beta, lambda)
        mask, gx, gy = rasterise_region(region)
        return mask.T, gy, gx
    return None


def _draw_sky_mask(ax, mask_tuple, mollweide, color, alpha=0.25, lw=1.1,
                   ls="solid", zorder=2):
    """Shade + outline a boolean sky mask in the chosen coordinates."""
    mask, lam, sinb = mask_tuple
    LAM, SB = np.meshgrid(lam, sinb)
    if mollweide:
        X, Y = LAM - np.pi, np.arcsin(np.clip(SB, -1.0, 1.0))
    else:
        X, Y = LAM, SB
    field = mask.astype(float)
    ax.contourf(X, Y, field, levels=[0.5, 1.5], colors=[color],
                alpha=alpha, zorder=zorder)
    ax.contour(X, Y, field, levels=[0.5], colors=[color], linewidths=lw,
               linestyles=ls, zorder=zorder + 0.1)


def _mask_extent(mask_tuple):
    """Native (lam_lo, lam_hi, sb_lo, sb_hi) enclosing a boolean mask, or None."""
    mask, lam, sinb = mask_tuple
    cols, rows = np.any(mask, axis=0), np.any(mask, axis=1)
    if not (cols.any() and rows.any()):
        return None
    return (lam[cols].min(), lam[cols].max(), sinb[rows].min(), sinb[rows].max())


# ---------------------------------------------------------------------------
# MCMC sky posterior (KDE on the native full-sky grid)
# ---------------------------------------------------------------------------

def _mcmc_sky_density(mcmc_file, ngrid, cr_area):
    """KDE the MCMC sky posterior on the native full-sky grid.

    Returns ``((LAM, SB, dens, area_cr), kde)`` where LAM/SB are meshgrids in
    native (lambda, sin beta), *dens* is normalised, *area_cr* is the sky area
    (sq deg) enclosed by the *cr_area* credible contour, and *kde* is the
    fitted :class:`gaussian_kde` (so the zoom figure can re-evaluate it on a
    finer grid).  ``(None, None)`` if the file lacks the sky columns.
    """
    samples, names = load_mcmc_samples(mcmc_file)
    if "lambda" not in names or "sinbeta" not in names:
        print(f"[warn] {mcmc_file} lacks lambda/beta columns; skipping MCMC.")
        return None, None
    lam_s = samples[:, names.index("lambda")]
    sinb_s = samples[:, names.index("sinbeta")]           # stored value is sin(beta)

    lam_g = np.linspace(0.0, 2.0 * np.pi, ngrid)
    sb_g = np.linspace(-1.0, 1.0, ngrid)
    LAM, SB = np.meshgrid(lam_g, sb_g)
    dlam = lam_g[1] - lam_g[0]
    dsb = sb_g[1] - sb_g[0]

    kde = gaussian_kde(np.vstack([lam_s, sinb_s]))
    dens = kde(np.vstack([LAM.ravel(), SB.ravel()])).reshape(LAM.shape)
    dens /= np.sum(dens) * dlam * dsb

    lvls, _ = contour_levels(dens, targets=(cr_area,))
    area_cr = compute_sky_area(dens, lvls[0], dlam, dsb)
    return (LAM, SB, dens, area_cr), kde


def _levels_native(dens, targets):
    """contour_levels with a strictly-increasing, de-duplicated level list."""
    lvls, _ = contour_levels(dens, targets=targets)
    return np.unique(lvls)


def _levels_styles(dens, targets, dashed_at):
    """Sorted, de-duplicated contour levels + a matching linestyle list.

    Levels whose target equals *dashed_at* are drawn ``"dashed"``, the rest
    ``"solid"``.  ``contour_levels`` returns levels ascending (widest CR first)
    with targets aligned, so the styles stay aligned after de-duplication.
    """
    lvls, tgts = contour_levels(dens, targets=targets)
    keep = np.concatenate(([True], np.diff(lvls) > 0))
    lvls, tgts = lvls[keep], tgts[keep]
    styles = ["dashed" if abs(t - dashed_at) < 1e-9 else "solid" for t in tgts]
    return lvls, styles


def _pct(targets):
    """Format credible-level targets for a label, e.g. (0.5, 0.9) -> '50/90%'."""
    return "/".join(f"{t:.0%}"[:-1] for t in targets) + "%"


# ---------------------------------------------------------------------------
# Per-round evaluation (one model load per round — the minimum)
# ---------------------------------------------------------------------------

def _evaluate_one_round(rd, r_idx, dataloader, ngrid, cr_area, need_area=True):
    """Evaluate a single round's sky posterior (one model load — the minimum).

    Returns ``(rec, model, ctx, inj)``: ``rec`` is ``{"round", "box", "area"}``,
    plus ``"norm"``/``"gx"``/``"gy"`` when the round has a sky head (``ctx`` and
    ``inj`` are ``None`` otherwise). ``model`` is always returned so callers can
    decide whether to keep it alive or free it.
    """
    box = load_prior_box(rd, r_idx)          # native bounds; no model needed
    print(f"[round {r_idx}] loading model from {rd}")
    model = load_model(os.path.join(rd, "checkpoints"))
    keys = keys_for_model(model)

    sky_idx = next((s for s in (SKY, SKY[::-1])
                    if find_out_param_idx(model, s) is not None), None)
    if sky_idx is None:
        print(f"  round {r_idx}: no sky head — area skipped.")
        return {"round": r_idx, "box": box, "area": None}, model, None, None

    out_idx = find_out_param_idx(model, sky_idx)
    p0_key = keys[sky_idx[0]]

    norm2d, inj_r, gx, gy = eval_posterior_2d(
        model, dataloader, sky_idx, out_idx, ngrid_points=ngrid,
        bounds_0=box[p0_key], bounds_1=box[keys[sky_idx[1]]],
        keep_batch_dim=True,
    )
    ctx = {"sky_idx": sky_idx, "out_idx": out_idx, "p0_key": p0_key}

    # Normalise orientation so x=lambda (cols), y=sin beta (rows).
    if p0_key != "lambda":
        gx, gy = gy.T, gx.T
        norm2d = np.swapaxes(norm2d, 1, 2)
        inj_r = np.asarray(inj_r)[:, ::-1]
    norm = norm2d[0]

    rec = {"round": r_idx, "box": box, "norm": norm, "gx": gx, "gy": gy, "area": None}
    if need_area:
        dlam = gx[0, 1] - gx[0, 0]
        dsb = gy[1, 0] - gy[0, 0]
        lvls, _ = contour_levels(norm, targets=(cr_area,))
        rec["area"] = compute_sky_area(norm, lvls[0], dlam, dsb)

    inj = (float(inj_r[0, 0]), float(inj_r[0, 1]))
    return rec, model, ctx, inj


def _evaluate_rounds(round_dirs, dataloader, ngrid, cr_area):
    """Load each round once; return per-round records + the final posterior.

    Each record: ``{"round", "box", "area"}`` plus, when the round has a sky
    head, the normalised posterior ``"norm"`` and its meshgrids ``"gx"``,
    ``"gy"`` (kept per round so the truncation-scheme panels can draw contour
    lines; ngrid*ngrid floats per round is negligible).  ``final`` points to the
    last record that has a posterior.  The injection (lambda, sin beta) is
    captured once (identical across rounds).

    Also returns ``(final_model, final_ctx)``: the loaded model of the last
    sky-head round is kept alive (the previous one is freed each time a newer
    sky-head round appears, so exactly one model is resident and no extra disk
    read is incurred) so the zoom figure can re-evaluate on a finer grid.
    """
    records = []
    inj = None
    final_model = None
    final_ctx = None
    for r_idx, rd in enumerate(round_dirs, start=1):
        rec, model, ctx, inj_r = _evaluate_one_round(
            rd, r_idx, dataloader, ngrid, cr_area, need_area=True)
        records.append(rec)

        if ctx is None:
            del model
            continue

        # Keep this model as the running "final"; free the previous one.
        if final_model is not None:
            del final_model
        final_model = model
        final_ctx = ctx
        if inj is None:
            inj = inj_r

    final = next((r for r in reversed(records) if "norm" in r), None)
    if final is None and final_model is not None:
        del final_model
        final_model = None
    return records, final, inj, final_model, final_ctx


def _evaluate_final_round(round_dirs, dataloader, ngrid, cr_area):
    """Like :func:`_evaluate_rounds`, but loads/evaluates only the LAST round.

    Used by the ``--zoom-only`` fast path: ``final_posterior_zoom.pdf`` only
    ever reads the final round's posterior (re-evaluated on a finer grid), so
    looping over every earlier round's checkpoint (needed for the per-round
    sky-area history and the truncation-scheme panels) is pure waste when
    that's the only figure being regenerated.
    """
    r_idx = len(round_dirs)
    rec, model, ctx, inj = _evaluate_one_round(
        round_dirs[-1], r_idx, dataloader, ngrid, cr_area, need_area=False)
    if ctx is None:
        raise RuntimeError(f"round {r_idx}'s model has no sky head.")
    return rec, inj, model, ctx


# ---------------------------------------------------------------------------
# Figure 1 — final posterior vs MCMC
# ---------------------------------------------------------------------------

def _fig_final_vs_mcmc(final, inj, mcmc, cr_area, outpath,
                       width_pt=None, fontsize=None):
    """``width_pt``/``fontsize`` given together switch to paper mode: serif/cm
    mathtext at a uniform fontsize, thinner lines, sparser graticule, no title
    (a paper caption covers that) -- default (both None) is the original bare
    10x6in canvas at matplotlib's ambient style.
    """
    paper_mode = width_pt is not None and fontsize is not None
    width_in = width_pt / PT_PER_INCH if paper_mode else _FINAL_FIG_WIDTH_IN
    height_in = width_in * (_FINAL_FIG_HEIGHT_IN / _FINAL_FIG_WIDTH_IN)
    contour_lw = 1.0 if paper_mode else 1.5
    inj_ms, inj_mew = (6, 1.0) if paper_mode else (12, 2)
    legend_fs = fontsize if paper_mode else 9
    rc = {
        "font.size": fontsize, "axes.labelsize": fontsize, "axes.titlesize": fontsize,
        "xtick.labelsize": fontsize, "ytick.labelsize": fontsize,
        "font.family": "serif", "mathtext.fontset": "cm",
    } if paper_mode else {}

    with plt.rc_context(rc):
        fig = plt.figure(figsize=(width_in, height_in))
        ax = fig.add_subplot(111, projection="mollweide")
        ax.grid(True, alpha=0.3)
        if paper_mode:
            # Sparser graticule labels -- the default ~11 longitude + ~11
            # latitude ticks collide at a 246pt paper column width.
            ax.set_xticks(np.radians([-120, -60, 0, 60, 120]))
            ax.set_yticks(np.radians([-60, -30, 0, 30, 60]))

        lon, lat = to_mollweide_coords(final["gx"], final["gy"], "lambda", "sinbeta")
        nre_lvls = _levels_native(final["norm"], _CR_CONTOURS)
        ax.contour(lon, lat, final["norm"], levels=nre_lvls,
                   colors="blue", linewidths=contour_lw, zorder=4)

        handles = [Line2D([0], [0], color="blue", label="NRE (final round)")]
        title = f"Sky posterior — round {final['round']}"
        if mcmc is not None:
            LAM, SB, dens, area_mcmc = mcmc
            mlon, mlat = to_mollweide_coords(LAM, SB, "lambda", "sinbeta")
            ax.contour(mlon, mlat, dens, levels=_levels_native(dens, _CR_CONTOURS),
                       colors="cyan", linewidths=contour_lw, zorder=3)
            handles.append(Line2D([0], [0], color="cyan", label="MCMC"))
            title += f"   (MCMC {cr_area:.0%} area = {area_mcmc:.1f} sq deg)"

        if inj is not None:
            ax.plot(inj[0] - np.pi, np.arcsin(np.clip(inj[1], -1, 1)),
                    "r+", markersize=inj_ms, markeredgewidth=inj_mew, zorder=5,
                    label="injection")
            handles.append(Line2D([0], [0], color="red", marker="+", ls="none",
                                  label="injection"))

        if not paper_mode:
            ax.set_title(title)
        ax.legend(handles=handles, loc="lower right", fontsize=legend_fs)
        fig.tight_layout()
        fig.savefig(outpath, bbox_inches="tight")
        plt.close(fig)
    print(f"wrote {outpath}")


# ---------------------------------------------------------------------------
# Figure 2 — sky area vs round
# ---------------------------------------------------------------------------

def _fig_area_vs_round(records, mcmc, cr_area, outpath):
    rounds = [r["round"] for r in records if r["area"] is not None]
    areas = [r["area"] for r in records if r["area"] is not None]
    if not rounds:
        print("[warn] no rounds with a sky head; skipping area-vs-round figure.")
        return
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(rounds, areas, "o-", color="C0", label="NRE")
    if mcmc is not None:
        ax.axhline(mcmc[3], ls="--", color="grey",
                   label=f"MCMC ({cr_area:.0%})")
    ax.set_yscale("log")
    ax.set_xlabel("training round")
    ax.set_ylabel(f"{cr_area:.0%} CR sky area [sq deg]")
    ax.set_title("Sky-localisation area across truncation rounds")
    ax.set_xticks(rounds)
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(outpath, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {outpath}")


# ---------------------------------------------------------------------------
# Figures 3 & 4 — truncation-scheme multi-panel
# ---------------------------------------------------------------------------

def _grid(n):
    cols = min(n, 5)
    rows = math.ceil(n / cols)
    return rows, cols


def _fig_truncation_scheme(records, inj, mollweide, outpath, name,
                           selected_rounds=None):
    """One panel per SELECTED round: the mask used as its training prior, the
    round's NRE sky posterior, and the mask proposed for the next round.

    ``selected_rounds`` is a set of 1-indexed round numbers (``None`` -> every
    round).  For round ``r`` the *prior* mask is the truncation saved at the end
    of round ``r-1`` and the *proposal* mask the one saved at the end of round
    ``r``.  Round 1's prior is the full box (no earlier mask); a missing mask
    (e.g. a rectangle-mode run, or the final round with no successor) falls back
    to that round's bounding box.
    """
    panels = ([r for r in records if r["round"] in selected_rounds]
              if selected_rounds is not None else list(records))
    if not panels:
        print("[warn] no rounds selected for the truncation-scheme figure.")
        return

    n = len(panels)
    rows, cols = _grid(n)
    fig = plt.figure(figsize=(4.2 * cols, 3.4 * rows))

    for i, rec in enumerate(panels):
        r = rec["round"]
        proj = "mollweide" if mollweide else None
        ax = fig.add_subplot(rows, cols, i + 1, projection=proj)

        cur = rec["box"]

        # (1) Prior mask that TRAINED this round = end-of-(r-1) truncation.
        prior_mask = _load_sky_mask(name, r - 1)
        if prior_mask is not None:
            _draw_sky_mask(ax, prior_mask, mollweide, color="0.35",
                           alpha=0.16, lw=1.0, zorder=2)
        else:
            # Round 1 (or rectangle run): the box itself is the prior.
            bx, by = _sky_box_loop(*cur["lambda"], *cur["sinbeta"], mollweide)
            ax.plot(bx, by, color="0.35", lw=1.0, ls="-", zorder=2)

        # (2) Proposal mask for the NEXT round = end-of-r truncation.
        prop_mask = _load_sky_mask(name, r)
        if prop_mask is not None:
            _draw_sky_mask(ax, prop_mask, mollweide, color="C1",
                           alpha=0.32, lw=1.3, zorder=3)
        else:
            nxt = next((rr["box"] for rr in records if rr["round"] == r + 1), cur)
            sx, sy = _sky_box_loop(*nxt["lambda"], *nxt["sinbeta"], mollweide)
            ax.fill(sx, sy, color="C1", alpha=0.32, zorder=3)

        # (3) NRE posterior contours (the density that motivated the proposal).
        if "norm" in rec:
            gx, gy = rec["gx"], rec["gy"]
            xx, yy = (to_mollweide_coords(gx, gy, "lambda", "sinbeta")
                      if mollweide else (gx, gy))
            ax.contour(xx, yy, rec["norm"],
                       levels=_levels_native(rec["norm"], _CR_CONTOURS),
                       colors="blue", linewidths=1.2, zorder=4)

        if inj is not None:
            if mollweide:
                ax.plot(inj[0] - np.pi, np.arcsin(np.clip(inj[1], -1, 1)),
                        "r+", markersize=9, markeredgewidth=2, zorder=5)
            else:
                ax.plot(inj[0], inj[1], "r+", markersize=9,
                        markeredgewidth=2, zorder=5)

        ax.set_title(f"Round {r}", fontsize=10)
        if mollweide:
            ax.grid(True, alpha=0.3)
        else:
            # Zoom to the prior-mask extent (with margin) so nesting is visible.
            ext = _mask_extent(prior_mask) if prior_mask is not None else None
            if ext is None:
                ext = (cur["lambda"][0], cur["lambda"][1],
                       cur["sinbeta"][0], cur["sinbeta"][1])
            lam_lo, lam_hi, sb_lo, sb_hi = ext
            mlam = 0.05 * (lam_hi - lam_lo) or 0.05
            msb = 0.05 * (sb_hi - sb_lo) or 0.05
            ax.set_xlim(lam_lo - mlam, lam_hi + mlam)
            ax.set_ylim(sb_lo - msb, sb_hi + msb)
            ax.set_xlabel(r"$\lambda$ [rad]")
            ax.set_ylabel(r"$\sin\beta$")
            ax.grid(True, alpha=0.3)

    space = "Mollweide $(\\lambda,\\ \\beta=\\arcsin\\sin\\beta)$" if mollweide \
        else r"native $(\lambda,\ \sin\beta)$"
    fig.suptitle(f"Sky truncation scheme — {space}", y=1.0)
    legend_handles = [
        Patch(facecolor="0.35", alpha=0.16, edgecolor="0.35",
              label="training-prior mask"),
        Patch(facecolor="C1", alpha=0.32, label="proposed next-round mask"),
        Line2D([0], [0], color="blue", label=f"NRE {_pct(_CR_CONTOURS)} CR"),
        Line2D([0], [0], color="red", marker="+", ls="none", label="injection"),
    ]
    fig.legend(handles=legend_handles, loc="lower center", ncol=4,
               fontsize=9, frameon=False, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout()
    fig.savefig(outpath, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {outpath}")


# ---------------------------------------------------------------------------
# Figure 5 — zoomed final-round posterior (Cartesian, native coords)
# ---------------------------------------------------------------------------

def _credible_bbox(norm, gx, gy, level, pad=0.25):
    """Bounding box (lam_lo, lam_hi, sb_lo, sb_hi) of the region above *level*,
    padded by a *pad* fraction of each side; falls back to the full grid."""
    lam, sb = gx[0, :], gy[:, 0]
    mask = norm >= level
    if not mask.any():
        return lam.min(), lam.max(), sb.min(), sb.max()
    cols, rows = np.any(mask, axis=0), np.any(mask, axis=1)
    lam_lo, lam_hi = lam[cols].min(), lam[cols].max()
    sb_lo, sb_hi = sb[rows].min(), sb[rows].max()
    mlam = pad * (lam_hi - lam_lo) or 0.01
    msb = pad * (sb_hi - sb_lo) or 0.01
    return lam_lo - mlam, lam_hi + mlam, sb_lo - msb, sb_hi + msb


def _compute_final_zoom(final, mcmc_kde, cr_area, model, dataloader, ctx, ngrid_zoom,
                        include=None):
    """The expensive part of the zoom figure: re-evaluate the NRE posterior on
    a fine grid restricted to the credible-region bounding box (a model
    forward pass over ``ngrid_zoom**2`` points) plus the MCMC KDE on the same
    grid. Returns everything :func:`_plot_final_zoom` needs, so pure styling
    tweaks (legend placement, fontsize, line width, ...) can be iterated on
    via :func:`_load_zoom_cache` without touching the model again.
    """
    coarse = final["norm"]
    lvl, _ = contour_levels(coarse, targets=(cr_area,))
    lam_lo, lam_hi, sb_lo, sb_hi = _credible_bbox(
        coarse, final["gx"], final["gy"], lvl[0])
    if include is not None:
        # widen the window to contain ``include`` = ((lam_lo, lam_hi), (sb_lo, sb_hi));
        # a wrapped axis (lo > hi) is left alone
        (ilo, ihi), (jlo, jhi) = include
        if ilo <= ihi:
            lam_lo, lam_hi = min(lam_lo, ilo), max(lam_hi, ihi)
        if jlo <= jhi:
            sb_lo, sb_hi = min(sb_lo, jlo), max(sb_hi, jhi)
    # never look outside the round's prior box (the padding can overshoot it)
    box = final.get("box") or {}
    if "lambda" in box:
        lam_lo, lam_hi = max(lam_lo, box["lambda"][0]), min(lam_hi, box["lambda"][1])
    if "sinbeta" in box:
        sb_lo, sb_hi = max(sb_lo, box["sinbeta"][0]), min(sb_hi, box["sinbeta"][1])

    sky_idx, p0_key = ctx["sky_idx"], ctx["p0_key"]
    if p0_key == "lambda":
        b0, b1 = (lam_lo, lam_hi), (sb_lo, sb_hi)
    else:
        b0, b1 = (sb_lo, sb_hi), (lam_lo, lam_hi)
    norm2d, _, gx, gy = eval_posterior_2d(
        model, dataloader, sky_idx, ctx["out_idx"], ngrid_points=ngrid_zoom,
        bounds_0=b0, bounds_1=b1, keep_batch_dim=True,
    )
    if p0_key != "lambda":
        gx, gy = gy.T, gx.T
        norm2d = np.swapaxes(norm2d, 1, 2)
    norm = norm2d[0]

    dens = None
    if mcmc_kde is not None:
        lam_g = np.linspace(lam_lo, lam_hi, ngrid_zoom)
        sb_g = np.linspace(sb_lo, sb_hi, ngrid_zoom)
        LAM, SB = np.meshgrid(lam_g, sb_g)
        dens = mcmc_kde(np.vstack([LAM.ravel(), SB.ravel()])).reshape(LAM.shape)

    return {"gx": gx, "gy": gy, "norm": norm, "dens": dens,
            "lam_lo": lam_lo, "lam_hi": lam_hi, "sb_lo": sb_lo, "sb_hi": sb_hi}


def _zoom_cache_path(outdir):
    return os.path.join(outdir, ".final_posterior_zoom_cache.npz")


def _save_zoom_cache(outdir, computed, inj, round_number, ngrid_zoom):
    kwargs = dict(round=round_number, ngrid_zoom=ngrid_zoom,
                  gx=computed["gx"], gy=computed["gy"], norm=computed["norm"],
                  lam_lo=computed["lam_lo"], lam_hi=computed["lam_hi"],
                  sb_lo=computed["sb_lo"], sb_hi=computed["sb_hi"],
                  inj=np.asarray(inj), has_dens=computed["dens"] is not None)
    if computed["dens"] is not None:
        kwargs["dens"] = computed["dens"]
    np.savez(_zoom_cache_path(outdir), **kwargs)


def _load_zoom_cache(outdir, round_number, ngrid_zoom):
    """Cached :func:`_compute_final_zoom` output, or None if missing/stale.

    Only ``round`` and ``ngrid_zoom`` are checked -- if ``--cr-area`` or
    ``--mcmc-file`` change between runs, pass ``--recompute-zoom`` to bypass.
    """
    path = _zoom_cache_path(outdir)
    if not os.path.exists(path):
        return None
    with np.load(path) as z:
        if int(z["round"]) != round_number or int(z["ngrid_zoom"]) != ngrid_zoom:
            return None
        computed = {
            "gx": z["gx"], "gy": z["gy"], "norm": z["norm"],
            "lam_lo": float(z["lam_lo"]), "lam_hi": float(z["lam_hi"]),
            "sb_lo": float(z["sb_lo"]), "sb_hi": float(z["sb_hi"]),
            "dens": z["dens"] if bool(z["has_dens"]) else None,
        }
        inj = tuple(z["inj"])
    return computed, inj


def _plot_final_zoom(computed, inj, cr_area, outpath, coords="native",
                     fontsize=_ZOOM_FONTSIZE, figsize=None, lw=4.5, region=None):
    """Pure plotting/styling from a :func:`_compute_final_zoom` result -- no
    model or dataloader touched, so this is fast to re-run for style-only
    tweaks (legend placement, fontsize, line width, ...).

    ``coords="native"`` plots (lambda [rad], sin beta); ``coords="lonlat"``
    remaps the same contours to ecliptic longitude / latitude in degrees
    (lon = degrees(lambda), lat = degrees(arcsin(sin beta))). The contour
    levels stay those of the native uniform grid, so the credible regions are
    identical -- only the axes change.

    Defaults are the poster styling; ``fontsize``/``figsize``/``lw`` override it.
    ``region`` (the next-round proposal, a MultiRegion) is drawn as a contour.
    """
    gx, gy, norm, dens = computed["gx"], computed["gy"], computed["norm"], computed["dens"]
    lam_lo, lam_hi = computed["lam_lo"], computed["lam_hi"]
    sb_lo, sb_hi = computed["sb_lo"], computed["sb_hi"]

    if coords == "lonlat":
        tx = lambda lam: np.degrees(np.asarray(lam, dtype=float))
        ty = lambda sb: np.degrees(np.arcsin(np.clip(sb, -1.0, 1.0)))
        xlabel, ylabel = r"$\lambda$ [deg]", r"$\beta$ [deg]"
    else:
        tx = ty = lambda v: v
        xlabel, ylabel = r"$\lambda$ [rad]", r"$\sin\beta$"

    with plt.rc_context({
        "font.size": fontsize,
        "axes.labelsize": fontsize,
        "axes.titlesize": fontsize,
        "xtick.labelsize": fontsize,
        "ytick.labelsize": fontsize,
        "font.family": "serif",
        "mathtext.fontset": "cm",
    }):
        fig, ax = plt.subplots(figsize=figsize or (_ZOOM_FIG_WIDTH_IN, _ZOOM_FIG_HEIGHT_IN * 1.05 * 0.95 * 0.9))

        lv, st = _levels_styles(norm, _CR_ZOOM, _CR_ZOOM_DASHED)
        ax.contour(tx(gx), ty(gy), norm, levels=lv, colors="blue",
                   linewidths=lw, linestyles=st, zorder=4)
        handles = [Line2D([0], [0], color="blue", lw=lw,
                          label=f"NRE {_pct(_CR_ZOOM)} CR")]

        if dens is not None:
            lam_g = np.linspace(lam_lo, lam_hi, dens.shape[1])
            sb_g = np.linspace(sb_lo, sb_hi, dens.shape[0])
            LAM, SB = np.meshgrid(lam_g, sb_g)
            lvm, stm = _levels_styles(dens, _CR_ZOOM, _CR_ZOOM_DASHED)
            ax.contour(tx(LAM), ty(SB), dens, levels=lvm, colors="cyan",
                       linewidths=lw, linestyles=stm, zorder=3)
            handles.append(Line2D([0], [0], color="cyan", lw=lw,
                                  label=f"MCMC {_pct(_CR_ZOOM)} CR"))

        if region is not None:
            acc = region.contains_grid((gx[0, :], gy[:, 0])).astype(float)
            if acc.any() and not acc.all():
                ax.contour(tx(gx), ty(gy), acc, levels=[0.5], colors="C2",
                           linewidths=lw, linestyles="dotted", zorder=4)
            handles.append(Line2D([0], [0], color="C2", lw=lw, ls="dotted",
                                  label="next-round proposal"))

        if inj is not None:
            ax.plot(tx(inj[0]), ty(inj[1]), "r+", markersize=4.4 * lw,
                    markeredgewidth=lw, zorder=5)
            handles.append(Line2D([0], [0], color="red", marker="+", ls="none",
                                  markersize=4.4 * lw, markeredgewidth=lw,
                                  label="injection"))

        ax.set_xlim(tx(lam_lo), tx(lam_hi))
        ax.set_ylim(ty(sb_lo), ty(sb_hi))
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)
        ax.legend(handles=handles, loc="center left", bbox_to_anchor=(1.02, 0.5),
                 ncol=1, fontsize=fontsize, frameon=False,
                 handlelength=1.2, handletextpad=0.4)
        fig.tight_layout()
        fig.savefig(outpath, bbox_inches="tight")
        plt.close(fig)
    print(f"wrote {outpath}")


def _fig_final_zoom(final, inj, mcmc_kde, cr_area, outpath,
                    model, dataloader, ctx, ngrid_zoom):
    """Zoomed final-round posterior: NRE re-evaluated from scratch on a finer
    grid over the credible region, plus a dashed 99% contour. Thin wrapper
    around compute+plot for the full-pipeline call site, which has no need
    for the cache (it already loads each round's model exactly once).
    """
    computed = _compute_final_zoom(final, mcmc_kde, cr_area, model, dataloader,
                                   ctx, ngrid_zoom)
    _plot_final_zoom(computed, inj, cr_area, outpath)
    _plot_final_zoom(computed, inj, cr_area,
                     outpath.replace(".pdf", "_lonlat.pdf"), coords="lonlat")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(
        description="Sky-marginal visualisation across TMNRE truncation rounds.",
    )
    p.add_argument("name", help="Run name / TIME_OF_EXECUTION.")
    p.add_argument("--data-path", default=None,
                   help="Observation HDF5 (default: the run\'s observation_used.yaml).")
    p.add_argument("--mcmc-file", default=None,
                   help="Flat MCMC samples HDF5 (for the MCMC overlay + area "
                        "baseline). Figs 3/4 work without it.")
    p.add_argument("--ngrid", type=int, default=200,
                   help="Grid resolution for posterior / KDE evaluation.")
    p.add_argument("--ngrid-zoom", type=int, default=300,
                   help="Finer grid resolution for the from-scratch zoom "
                        "re-evaluation of the final posterior (default 300).")
    p.add_argument("--rounds", type=int, nargs="+", default=None,
                   help="1-indexed round numbers to draw as per-round sky panels "
                        "(prior mask + posterior + proposal mask). Only these "
                        "rounds are shown in the truncation-scheme figures; "
                        "omit to show every round.")
    p.add_argument("--last-round", type=int, default=None,
                   help="Stop at this round (1-indexed, inclusive).")
    p.add_argument("--ckpt-final-round", default=None,
                   help="Checkpoint overriding the final round's truncation.ckpt.")
    p.add_argument("--cr-area", type=float, default=0.90,
                   help="Credible level for the sky-area metric (default 0.90).")
    p.add_argument("--zoom-only", action="store_true",
                   help="Only regenerate final_posterior_zoom.pdf, loading just "
                        "the final round's checkpoint instead of every round's "
                        "(skips the other 4 figures, which need the full "
                        "per-round history). Caches the fine-grid evaluation "
                        "(the expensive part) so subsequent --zoom-only runs "
                        "that only change plot styling skip the model entirely.")
    p.add_argument("--recompute-zoom", action="store_true",
                   help="With --zoom-only, ignore any cached fine-grid "
                        "evaluation and recompute from the model (needed if "
                        "--cr-area or --mcmc-file changed since the cache was "
                        "written).")
    p.add_argument("--width-pt", type=float, default=None,
                   help="final_posterior_mcmc.pdf width in points -- pass "
                        "together with --fontsize to switch it to paper mode "
                        "(serif/cm mathtext, thinner lines, no title). E.g. "
                        "246 for a paper single-column figure. Does not affect "
                        "any of the other 4 figures this script produces.")
    p.add_argument("--fontsize", type=float, default=None,
                   help="final_posterior_mcmc.pdf uniform fontsize -- see "
                        "--width-pt.")
    p.add_argument("--final-only", action="store_true",
                   help="Only regenerate final_posterior_mcmc.pdf, loading just "
                        "the final round's checkpoint instead of every round's "
                        "(skips the other 4 figures, which need the full "
                        "per-round history -- same idea as --zoom-only).")
    args = p.parse_args()

    round_dirs = find_round_dirs(args.name)
    if not round_dirs:
        raise RuntimeError(f"No round directories found for name='{args.name}'.")
    print(f"Detected {len(round_dirs)} round(s) for '{args.name}'.")

    if args.last_round is not None:
        if not 1 <= args.last_round <= len(round_dirs):
            raise ValueError(f"--last-round={args.last_round} out of range "
                             f"[1, {len(round_dirs)}]")
        round_dirs = round_dirs[: args.last_round]
        print(f"Truncated to first {args.last_round} round(s).")

    if args.ckpt_final_round:
        register_ckpt_override(round_dirs[-1], args.ckpt_final_round)

    dataloader = build_obs_dataloader(resolve_obs_path(args.name, args.data_path))

    outdir = os.path.join(
        PLOTS_ROOT_DIR,
        args.name + (f"_upto_round_{args.last_round}" if args.last_round else ""),
        "sky",
    )
    os.makedirs(outdir, exist_ok=True)

    if args.zoom_only:
        round_number = len(round_dirs)
        cached = None if args.recompute_zoom else _load_zoom_cache(
            outdir, round_number, args.ngrid_zoom)
        if cached is not None:
            print(f"[zoom-only] using cached fine-grid evaluation "
                  f"({_zoom_cache_path(outdir)}); no model loaded.")
            computed, inj = cached
        else:
            final, inj, final_model, final_ctx = _evaluate_final_round(
                round_dirs, dataloader, args.ngrid, args.cr_area,
            )
            mcmc_kde = None
            if args.mcmc_file:
                _, mcmc_kde = _mcmc_sky_density(args.mcmc_file, args.ngrid, args.cr_area)
            computed = _compute_final_zoom(final, mcmc_kde, args.cr_area,
                                           final_model, dataloader, final_ctx,
                                           args.ngrid_zoom)
            _save_zoom_cache(outdir, computed, inj, round_number, args.ngrid_zoom)
        _plot_final_zoom(computed, inj, args.cr_area,
                         os.path.join(outdir, "final_posterior_zoom.pdf"))
        _plot_final_zoom(computed, inj, args.cr_area,
                         os.path.join(outdir, "final_posterior_zoom_lonlat.pdf"),
                         coords="lonlat")
        return

    if args.final_only:
        final, inj, final_model, final_ctx = _evaluate_final_round(
            round_dirs, dataloader, args.ngrid, args.cr_area,
        )
        del final_model  # not needed -- only the posterior/inj already evaluated
        mcmc = None
        if args.mcmc_file:
            mcmc, _ = _mcmc_sky_density(args.mcmc_file, args.ngrid, args.cr_area)
        _fig_final_vs_mcmc(final, inj, mcmc, args.cr_area,
                           os.path.join(outdir, "final_posterior_mcmc.pdf"),
                           width_pt=args.width_pt, fontsize=args.fontsize)
        return

    records, final, inj, final_model, final_ctx = _evaluate_rounds(
        round_dirs, dataloader, args.ngrid, args.cr_area,
    )

    mcmc, mcmc_kde = None, None
    if args.mcmc_file:
        mcmc, mcmc_kde = _mcmc_sky_density(args.mcmc_file, args.ngrid, args.cr_area)

    if final is not None:
        _fig_final_vs_mcmc(final, inj, mcmc, args.cr_area,
                           os.path.join(outdir, "final_posterior_mcmc.pdf"),
                           width_pt=args.width_pt, fontsize=args.fontsize)
        _fig_final_zoom(final, inj, mcmc_kde, args.cr_area,
                        os.path.join(outdir, "final_posterior_zoom.pdf"),
                        final_model, dataloader, final_ctx, args.ngrid_zoom)
    else:
        print("[warn] no round had a sky head; skipping final-posterior figure.")

    _fig_area_vs_round(records, mcmc, args.cr_area,
                       os.path.join(outdir, "sky_area_vs_round.pdf"))

    selected_rounds = None
    if args.rounds is not None:
        avail = {r["round"] for r in records}
        selected_rounds = sorted(set(args.rounds))
        missing = [r for r in selected_rounds if r not in avail]
        if missing:
            raise ValueError(f"--rounds {missing} not among available rounds "
                             f"{sorted(avail)}.")
        no_post = [r for r in selected_rounds
                   if "norm" not in next(x for x in records if x["round"] == r)]
        if no_post:
            print(f"[warn] rounds {no_post} have no sky head; their panels show "
                  f"masks only (no posterior contour).")
        selected_rounds = set(selected_rounds)
        print(f"Per-round sky panels for rounds: {sorted(selected_rounds)}")

    _fig_truncation_scheme(records, inj, True,
                           os.path.join(outdir, "truncation_scheme_mollweide.pdf"),
                           args.name, selected_rounds)
    _fig_truncation_scheme(records, inj, False,
                           os.path.join(outdir, "truncation_scheme_native.pdf"),
                           args.name, selected_rounds)


if __name__ == "__main__":
    main()
