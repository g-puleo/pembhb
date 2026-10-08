"""Unit tests for the _StepCadence gate (step-based callback cadence).

Proves the gate fires on an exact global-step cadence, independent of where
epoch boundaries fall, and is disabled (no-op) when interval=None.
"""

from pembhb.callbacks import _StepCadence


def test_fires_on_exact_step_cadence():
    """Driven per-step (as on_train_batch_end sees), fire at the first call then
    every >= interval steps. With global_step starting at 0 and interval 1000,
    that is 0, 1000, 2000, 3000 — regardless of interleaved epoch boundaries."""
    gate = _StepCadence(1000)
    fired = []
    epoch_len = 80  # small "streaming" epoch; boundaries must not affect cadence
    for step in range(0, 3001):
        if step % epoch_len == 0:
            pass  # pretend an epoch boundary happened here — gate must ignore it
        if gate.should_fire(step):
            fired.append(step)
    assert fired == [0, 1000, 2000, 3000], fired


def test_gaps_never_below_interval():
    gate = _StepCadence(500)
    last = None
    for step in range(0, 5000):
        if gate.should_fire(step):
            if last is not None:
                assert step - last >= 500
            last = step


def test_disabled_when_interval_none():
    gate = _StepCadence(None)
    assert gate.enabled is False


class _FakePlot:
    """Minimal stand-in for PlotPosteriorCallback (only needs the entry dicts)."""
    def __init__(self):
        self.volume_ratios = {}
        self.differential_entropies = {}


def test_es_consumes_each_plot_step_once():
    """The step-mode early-stop must consume the newest plot-cb entries exactly
    once per new global step (so EMA/stall advance once per plot fire, not per
    training step)."""
    from pembhb.callbacks import VolumeRatioEarlyStopping
    es = VolumeRatioEarlyStopping(step_mode=True)
    fp = _FakePlot()
    fp.volume_ratios = {(0,): [{'step': 100, 'ratio': 0.9}],
                        (1,): [{'step': 100, 'ratio': 0.8}]}
    assert es._collect_current_by_step(fp) == {(0,): 0.9, (1,): 0.8}
    # nothing new -> empty (no double-counting between plot fires)
    assert es._collect_current_by_step(fp) == {}
    # next plot fire at step 200
    fp.volume_ratios[(0,)].append({'step': 200, 'ratio': 0.5})
    fp.volume_ratios[(1,)].append({'step': 200, 'ratio': 0.4})
    assert es._collect_current_by_step(fp) == {(0,): 0.5, (1,): 0.4}
