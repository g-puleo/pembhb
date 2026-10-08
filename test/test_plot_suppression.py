"""§2 plot awareness: PlotPosteriorCallback shades the region outside the
previous round's trained mask (where the density is zeroed / down-weighted)."""
import numpy as np
import pytest

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pembhb.callbacks import PlotPosteriorCallback
from pembhb.regions import Region


def _cb(marginal, prev_accepted, zero_policy):
    return PlotPosteriorCallback(
        timestamp="t", obs_loader=None, input_idx_list=[list(marginal)],
        output_idx_list=[0], round_idx=2,
        prev_accepted=prev_accepted, zero_policy=zero_policy)


def test_shade_1d_adds_span_when_active():
    cb = _cb((6,), {(6,): {"kind": "1d", "intervals": [[-0.5, 0.5]]}}, "hard")
    fig, ax = plt.subplots()
    cb._shade_suppressed_1d(ax, np.linspace(-1, 1, 100), (6,))
    # two suppressed runs ([-1,-0.5) and (0.5,1]) -> two axvspan polygons
    assert len(ax.patches) == 2
    # exactly one carries the legend label (labelled once)
    labels = [p.get_label() for p in ax.patches if p.get_label() and not p.get_label().startswith("_")]
    assert len(labels) == 1 and "§2" in labels[0]
    plt.close(fig)


def test_shade_1d_off_and_round1_are_noops():
    fig, ax = plt.subplots()
    # policy off
    _cb((6,), {(6,): {"kind": "1d", "intervals": [[-0.5, 0.5]]}}, "off") \
        ._shade_suppressed_1d(ax, np.linspace(-1, 1, 100), (6,))
    assert len(ax.patches) == 0
    # round 1: no previous mask for this marginal
    _cb((6,), {}, "hard")._shade_suppressed_1d(ax, np.linspace(-1, 1, 100), (6,))
    assert len(ax.patches) == 0
    plt.close(fig)


def test_shade_1d_hysteresis_label():
    cb = _cb((6,), {(6,): {"kind": "1d", "intervals": [[-0.5, 0.5]]}}, "hysteresis")
    fig, ax = plt.subplots()
    cb._shade_suppressed_1d(ax, np.linspace(-1, 1, 100), (6,))
    labels = [p.get_label() for p in ax.patches if p.get_label() and not p.get_label().startswith("_")]
    assert any("down-weighted" in l for l in labels)
    plt.close(fig)


def test_shade_2d_adds_image_when_active():
    g0 = np.linspace(0, 1, 60); g1 = np.linspace(0, 1, 40)
    gx, gy = np.meshgrid(g0, g1, indexing="xy")
    mask = np.zeros((40, 60), bool)
    mask[np.ix_((g1 >= 0.2) & (g1 <= 0.5), (g0 >= 0.3) & (g0 <= 0.6))] = True
    cb = _cb((7, 8), {(7, 8): {"kind": "2d", "region": Region(mask, (g0, g1))}}, "hard")
    fig, ax = plt.subplots()
    cb._shade_suppressed_2d(ax, gx, gy, (7, 8))
    assert len(ax.images) == 1                    # translucent wash over ~keep
    plt.close(fig)


def test_shade_2d_off_is_noop():
    g0 = np.linspace(0, 1, 60); g1 = np.linspace(0, 1, 40)
    gx, gy = np.meshgrid(g0, g1, indexing="xy")
    mask = np.zeros((40, 60), bool); mask[5:20, 10:30] = True
    cb = _cb((7, 8), {(7, 8): {"kind": "2d", "region": Region(mask, (g0, g1))}}, "off")
    fig, ax = plt.subplots()
    cb._shade_suppressed_2d(ax, gx, gy, (7, 8))
    assert len(ax.images) == 0
    plt.close(fig)
