"""Heterodyne bins for a round-trip datagen config, the binned observation and the matching config.

    python scripts/build_heterodyne.py --config configs/datagen_maskzero_rt_84d.yaml \
        --fiducial /data/gpuleo/mbhb/long_tobs_heterodyne/theta_med.yaml \
        --box /data/gpuleo/mbhb/2026/10/15/channelizedmlp_maskzero_mlp60k/prior_after_round_26.yaml \
        --obs /data/gpuleo/mbhb/obs_nested_rt/obs_84d_withnoise.h5 --n-bins 8000 --out-dir DIR

Bins: equal summed drift of h / h_fid at the box edges (heterodyne.drift). Writes DIR/het_{N}.npz,
DIR/{obs}_het{N}_withnoise.h5 (signal and stored noise of the fine obs, binned separately) and
configs/{config}_het.yaml: the same config with td.heterodyne.file and the box as prior.
"""
import argparse
import copy
import os

import h5py
import numpy as np
import yaml

from pembhb import ROOT_DIR
from pembhb.heterodyne import Heterodyne, bins_from_drift, drift, support_start
from pembhb.simulator import MBHBSimulatorFD
from pembhb.utils import ordered_prior_keys

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--config", required=True)
parser.add_argument("--fiducial", required=True, help="yaml with theta_med (fiducial) and theta_inj")
parser.add_argument("--box", required=True, help="prior yaml (key 'prior') the bins must cover")
parser.add_argument("--obs", required=True, help="fine-grid obs with stored noise_fd")
parser.add_argument("--n-bins", type=int, default=8000)
parser.add_argument("--out-dir", required=True)
args = parser.parse_args()

conf = yaml.safe_load(open(os.path.join(ROOT_DIR, args.config)))
if (conf["waveform_params"].get("td") or {}).get("heterodyne") is not None:
    raise ValueError(f"{args.config} is already heterodyned; pass the fine config")
fid = yaml.safe_load(open(args.fiducial))
theta_fid, theta_inj = fid["theta_med"], fid["theta_inj"]
box = {k: [float(v[0]), float(v[1])] for k, v in yaml.safe_load(open(args.box))["prior"].items()}

sim_conf = copy.deepcopy(conf)
sim_conf["prior"] = box
sim = MBHBSimulatorFD(sim_conf, sampler_init_kwargs={"prior_bounds": box,
                                                     "spin_param_basis": conf["spin_param_basis"]})
keys = ordered_prior_keys(sim.sampler.spin_param_basis)


def bbhx(theta):
    x = np.array([[float(theta[k])] for k in keys])
    return sim.sampler.samples_to_bbhx_input(x, sim.t_obs_end_SI)


def h_of(theta):
    return sim.waveform(bbhx(theta))[0].astype(np.complex128)


h_fid = h_of(theta_fid)
i0 = support_start(h_fid)
edges, eps = bins_from_drift(drift(h_of, theta_fid, box, h_fid, i0), args.n_bins)
w = 4 * sim.df / sim.asd ** 2
het = Heterodyne.from_fiducial(edges + i0, h_fid, w, sim.freqs, sim.df, eps)
os.makedirs(args.out_dir, exist_ok=True)
het_path = os.path.join(args.out_dir, f"het_{args.n_bins}.npz")
het.save(het_path)
print(f"{het.n_bins} bins from {sim.freqs[i0]:.3e} Hz (fine bin {i0}), eps {eps:.4f}, "
      f"fiducial SNR {np.sqrt(het.B.sum()):.1f} -> {het_path}")

het_conf = copy.deepcopy(conf)
het_conf["prior"] = box
het_conf["waveform_params"]["td"]["heterodyne"] = {"file": het_path}
conf_path = os.path.join(ROOT_DIR, args.config.removesuffix(".yaml") + "_het.yaml")
with open(conf_path, "w") as f:
    f.write(f"# {args.config} heterodyned on {het_path} (scripts/build_heterodyne.py); prior = {args.box}\n")
    yaml.safe_dump(het_conf, f, sort_keys=False)

het_sim = MBHBSimulatorFD(copy.deepcopy(het_conf), sampler_init_kwargs={
    "prior_bounds": box, "spin_param_basis": conf["spin_param_basis"]})
with h5py.File(args.obs) as f:
    f_obs = f["frequencies"][()]
    if f_obs.shape != sim.freqs.shape or not np.allclose(f_obs, sim.freqs, rtol=1e-6, atol=0):
        raise ValueError(f"{args.obs} is not on the fine grid of {args.config}")
    params, bb = f["source_parameters"][()].T, f["bbhx_parameters"][()].T
    h_obs, n_obs = het.bin(f["wave_fd"][()]), het.bin(f["noise_fd"][()])
    noise_seed = f.attrs.get("noise_seed", -1)

obs_path = os.path.join(args.out_dir, os.path.basename(args.obs).replace(
    "_withnoise.h5", f"_het{args.n_bins}_withnoise.h5"))
with h5py.File(obs_path, "w") as f:
    ds = het_sim._create_datasets(f, h_obs.shape[0], store_noise=True)
    het_sim._write_batch(ds, 0, params, bb, h_obs, n_obs)
    f.attrs["noise_seed"] = noise_seed
    f.attrs["fine_obs"] = args.obs
het_sim.save_info_yaml(filename=obs_path, overwrite=True)

B = het.B
chi2 = np.sum(np.abs(n_obs[0]) ** 2 * B)
dh = np.sum(np.abs(h_obs[0] - het_sim.waveform(bbhx(theta_inj))[0]) ** 2 * B)
print(f"obs: binned SNR {het_sim.get_SNR_FD(h_obs)[0]:.1f}, noise chi2 {chi2:.0f} vs {2 * B.size} "
      f"+- {np.sqrt(4 * B.size):.0f}, |r(h_obs) - r(h_sim(theta_inj))|^2 B = {dh:.3g} -> {obs_path}")
print(f"config -> {conf_path}")
