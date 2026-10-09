# Installation

`pembhb` generates waveforms with [`bbhx`](https://github.com/mikekatz04/BBHx)
and LISA noise and orbits with
[`lisaanalysistools`](https://github.com/mikekatz04/LISAanalysistools). Both are
installed from pip. The pinned versions below need **two small patches to bbhx**
before they work together.

Tested with Python 3.11, torch 2.6 (CUDA 12.4), and the package versions below:

| package | version |
|---|---|
| `bbhx` (+ `bbhx-cuda12x`) | 1.2.3 |
| `lisaanalysistools` (+ `lisaanalysistools-cuda12x`) | 1.2.8 |
| `gpubackendtools` | 0.1.1 |
| `h5py` | ≥ 3.5 (tested 3.16 / HDF5 2.0) |

## 1. Install the package

```bash
git clone https://github.com/g-puleo/pembhb.git
cd pembhb
python -m venv .venv && source .venv/bin/activate   # or conda
pip install torch                                     # pick the build matching your CUDA
pip install -e ".[cuda12x]"                           # GPU (recommended)
# pip install -e .                                    # CPU only
```

For CPU only, set `backend: "cpu"` in `configs/datagen_config.yaml`. Expect
simulation to be much slower.

## 2. Patch bbhx (required)

Apply both patches to the installed bbhx, and re-apply them whenever you
reinstall it:

```bash
SITE=$(python -c "import bbhx, os; print(os.path.dirname(os.path.dirname(bbhx.__file__)))")
patch -p1 -d "$SITE" < patches/bbhx-1.2.3-orbits-ptr.patch
patch -p1 -d "$SITE" < patches/bbhx-1.2.3-tstart-window.patch
```

### 2a. Orbit pointer (`bbhx-1.2.3-orbits-ptr.patch`)

**Symptom.** Every simulation call fails inside bbhx:

```
File ".../bbhx/response/fastfdresponse.py", line 460, in __call__
    self.response_gen(
  File "response.pyx", line 62, in response.LISA_response_wrap
TypeError: an integer is required
```

**Cause.** `LISAResponse.__call__` passes the orbit wrapper object
(`EqualArmlengthOrbits`) to the Cython function `LISA_response_wrap`, which
expects a `size_t`: the raw address of the C++ `Orbits` object. Normally
`gpubackendtools.pointeradjust.wrapper()` would convert it via
`obj.ptr`. In `lisaanalysistools` 1.2.8, however, the pybind11 class
`OrbitsWrapGPU`/`OrbitsWrapCPU` has no `.ptr`. The wrapper swallows the
resulting `AttributeError` and returns the Python object unchanged, and Cython
then fails to convert it to an integer.

**Fix.** The patch caches the C++ orbit object when the orbits are set, then
reads its pointer from the pybind11 instance layout. In pybind11's simple
layout the value pointer sits right after the 16-byte `PyObject` header. The
patch passes that integer instead of the wrapper. It works for both the CPU
and the CUDA backend.

The patch touches only `bbhx/response/fastfdresponse.py`, in three places:
`import ctypes`, pointer extraction in the `orbits` setter, and
`self._orbits_ptr` in place of `self.orbits` in the `response_gen(...)` call.
If you change the pybind11 version, re-check the 16-byte offset.

### 2b. Observation window start (`bbhx-1.2.3-tstart-window.patch`)

Only matters when `waveform_params.t_obs_start > 0`, i.e. the observation
window does not begin at LISA time 0.

**Symptom.** With `t_obs_start > 0` the waveform is identically zero (SNR 0).

**Cause.** bbhx builds the waveform on a sparse frequency grid, trimmed to the
frequencies emitted inside the observation window, then interpolates. The trim
in `PhenomHMAmpPhase.__call__` hardcodes the window as `0 < t(f) < Tobs`, while
`BBHWaveformFD` passes `Tobs = t_end - t_start`, a duration. For a window
`[t_start, t_end]` far from 0 the sparse grid describes the wrong stretch of
the inspiral and the response step then zeroes all of it.

**Fix.** `BBHWaveformFD` also passes `t_start`, and the trim becomes
`t_start < t(f) < t_start + Tobs`, with `Tobs` keeping its meaning of duration.
`t_start` is clamped at 0: the SSB-frame window start can be up to ~500 s
negative, where no orbit data exists. With the default `t_start = 0` the
behaviour is unchanged. Touches `bbhx/waveformbuild.py` (one argument) and
`bbhx/waveforms/phenomhm.py` (the two trims).

## 3. Check the install

```bash
pytest                       # ~30 s, CPU only, no data files needed
```

`test/test_simulator.py` builds bbhx waveforms on the CPU backend, so it fails
with the `TypeError` above if patch 2a is missing. The tests use
`t_obs_start = 0` and do not detect a missing patch 2b.

## 4. Where outputs go

| env var | default | contents |
|---|---|---|
| `PEMBHB_DATA_DIR` | `<repo>/data` | simulations, streaming buffers, TensorBoard logs, checkpoints |
| `PEMBHB_PLOTS_DIR` | `<repo>/plots` | training-time and visualisation figures |

Point both at a disk with room: a run writes several GB.

## Known environment issue

On HDF5 ≥ 2.0, file locking can raise `BlockingIOError: errno 11` with
`streaming.storage: disk`. The disk ring already opens its read handles with
`locking=False`. If you still see the error on your filesystem, export
`HDF5_USE_FILE_LOCKING=FALSE`.
