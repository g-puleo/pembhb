"""Nested observations: one long signal + noise stream, cut to several windows ending at the same t_obs_end.

Each --configs entry is a round-trip datagen config (waveform_params.td, taper > 0) that defines one cut:
[t_obs_start, t_obs_start + duration]. All must share t_obs_end, injection, dt, taper and channels.

    python scripts/build_nested_obs.py --configs configs/datagen_maskzero_rt_1w.yaml \
        configs/datagen_maskzero_rt_84d.yaml --out-dir /data/gpuleo/mbhb/obs_nested_rt

Writes obs_long.h5 (TD streams, transferable) and obs_{cut}d_withnoise.h5 (+ .yaml) per config, in the
format tmnre_joint.py --obs-path expects. The long stream is the injection generated over --long-days
(default: the longest duration + lookback) plus stationary noise drawn on that circle; the wrap-around of
the circle lies outside every cut.
"""
import argparse
import copy
import os

import h5py
import numpy as np
import yaml

from pembhb import ROOT_DIR
from pembhb.simulator import DAY_SI, MBHBSimulatorFD, build_asd

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--configs", nargs="+", required=True)
parser.add_argument("--out-dir", required=True)
parser.add_argument("--long-days", type=float, default=None)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--noise-seed", type=int, default=0)
args = parser.parse_args()

sims, confs = [], []
for path in args.configs:
    conf = yaml.safe_load(open(os.path.join(ROOT_DIR, path)))
    inj = conf["injection"]
    conf["prior"] = {k: [float(inj[k]), float(inj[k])] for k in conf["prior"]}
    td = conf["waveform_params"].get("td") or {}
    if not td.get("round_trip") or not td.get("taper_days"):
        raise ValueError(f"{path}: needs td.round_trip and taper_days > 0 (a rectangular cut of a long "
                         "stream does not have the diagonal noise the training assumes)")
    confs.append(conf)
    sims.append(MBHBSimulatorFD(conf, sampler_init_kwargs={"prior_bounds": conf["prior"],
                                                           "spin_param_basis": conf["spin_param_basis"]},
                                seed=args.seed))

ref = sims[0]
for s, c in zip(sims[1:], confs[1:]):
    assert abs(s.t_obs_end_SI - ref.t_obs_end_SI) < 1e-3, "all cuts must end at the same t_obs_end"
    assert s.dt == ref.dt and s.channels == ref.channels and c["injection"] == confs[0]["injection"]
    assert c["waveform_params"]["td"]["taper_days"] == confs[0]["waveform_params"]["td"]["taper_days"]

t_end, dt = ref.t_obs_end_SI, ref.dt
need = max(s.t_obs_end_SI - s.t_gen_start_SI for s in sims) / DAY_SI
long_days = args.long_days or need
if long_days < need:
    raise ValueError(f"--long-days {long_days} < longest duration + lookback = {need:.2f} d")
n_long = int(round(long_days * DAY_SI / dt))
t0 = t_end - n_long * dt
if t0 < 0:
    raise ValueError(f"stream would start at t = {t0 / DAY_SI:.1f} d < 0; shift t_obs_start by a year")

out = ref.sample(1)
params, bbhx = out["parameters"], out["bbhx_parameters"]
x = ref.td_signal(bbhx, t0)
signal_td = np.asarray(x.get() if hasattr(x, "get") else x)[0]

rng = np.random.default_rng(args.noise_seed)
f_long = np.fft.rfftfreq(n_long, dt)
sigma = np.zeros((ref.n_channels, f_long.size))
sigma[:, 1:] = build_asd(f_long[1:], ref.channels, confs[0]["waveform_params"]["noise"]) * np.sqrt(n_long * dt / 4)
z = rng.normal(size=sigma.shape) + 1j * rng.normal(size=sigma.shape)
noise_td = np.fft.irfft(z * sigma, n=n_long, axis=-1) / dt

os.makedirs(args.out_dir, exist_ok=True)
with h5py.File(os.path.join(args.out_dir, "obs_long.h5"), "w") as f:
    f.create_dataset("signal_td", data=signal_td)
    f.create_dataset("noise_td", data=noise_td)
    f.create_dataset("source_parameters", data=params[:, 0])
    f.create_dataset("bbhx_parameters", data=bbhx[:, 0])
    f.attrs.update(dt=dt, t0=t0, t_end=t_end, channels=ref.channels, noise_seed=args.noise_seed,
                   configs=[yaml.safe_dump(c) for c in confs])
print(f"long stream: {long_days:.1f} d, {n_long} samples, [{t0 / DAY_SI:.2f}, {t_end / DAY_SI:.2f}] d")

for s in sims:
    cut_days = s.obs_length / DAY_SI
    h = s.cut_to_fd(signal_td[None])
    n = s.cut_to_fd(noise_td[None])
    fname = os.path.join(args.out_dir, f"obs_{cut_days:g}d_withnoise.h5")
    with h5py.File(fname, "w") as f:
        ds = s._create_datasets(f, 1, store_noise=True)
        s._write_batch(ds, 0, params, bbhx, h, n)
        f.attrs["noise_seed"] = args.noise_seed
        f.attrs["long_stream"] = "obs_long.h5"
    s.save_info_yaml(filename=fname, overwrite=True)

    w = 4 * s.df / s.asd ** 2
    dh = np.sum(np.abs(h[0] - s.waveform(bbhx)[0]) ** 2 * w)
    chi2 = np.sum(np.abs(n[0]) ** 2 * w)        # Re and Im each N(0, 1) after whitening
    print(f"{cut_days:g} d: SNR {s.get_SNR_FD(h)[0]:.1f}, |h_obs - h_sim|^2 = {dh:.3g} (sim lookback "
          f"{s.td['lookback_days']} d), whitened noise chi2/dof = {chi2 / (2 * h.shape[1] * h.shape[2]):.3f} "
          f"-> {fname}")
