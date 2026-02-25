import argparse
import torch
from pembhb.data import MBHBDataset, MBHBDataModule
from pembhb.rom import ReducedOrderModel
from pembhb.utils import read_config, ROOT_DIR
import os

def build_rom(dataloader, outfile, tolerance, domain='fd', convergence_on='sigma_data',
              use_pinned_memory=False, prefetch_batches=1, freq_cutoff_idx=None, df=None):
    rom = ReducedOrderModel(tolerance=tolerance, device="cuda", debugging=False, domain=domain, freq_cutoff_idx=freq_cutoff_idx, df=df)
    rom.train(dataloader, use_pinned_memory=use_pinned_memory, prefetch_batches=prefetch_batches,
              convergence_on=convergence_on)
    rom.to_file(outfile)
    

# example usage
if __name__ == "__main__":
    # you already have a dataset object with dataset.wave_fd

    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--out", type=str, required=True)
    parser.add_argument("--tol", type=float)
    parser.add_argument("--noise-factor", type=float, default=1.0)
    parser.add_argument("--bs", type=int, default=1000)
    parser.add_argument("--domain", type=str, default="fd", choices=["fd", "td", "both"],
                        help="Which representation(s) to build a basis for: fd, td, or both")
    parser.add_argument("--convergence-on", type=str, default="sigma_data",
                        choices=["sigma", "sigma_data"],
                        help="Convergence criterion: 'sigma' (normalised waveform) or 'sigma_data' (noisy data)")
    parser.add_argument("--gpu-data", action="store_true", help="Move entire dataset to GPU (faster but uses more VRAM)")
    parser.add_argument("--pin-memory", action="store_true", help="Use pinned CPU memory (faster transfers but locks RAM)")
    parser.add_argument("--prefetch-batches", type=int, default=1, help="Concatenate this many batches before processing")
    parser.add_argument("--patience", type=int, default=10, help="Number of epochs to wait for improvement before early stopping")
    parser.add_argument("--freq-cutoff-idx", type=int, default=None, help="Discard fd bins below this index")
    args = parser.parse_args()
    
    train_conf = read_config(os.path.join(ROOT_DIR, "configs", "train_config.yaml"))
    batch_size = train_conf["batch_size"]
    dmodule = MBHBDataModule(args.data, batch_size, num_workers=0, cache_in_memory=True, noise_factor=args.noise_factor)
    freqs = dmodule.get_freqs()
    df = (freqs[1] - freqs[0]).item()
    dmodule.setup("fit")
    
    # --pin-memory enables pinned memory; --gpu-data moves all to GPU (mutually exclusive in effect)
    use_pinned_memory = args.pin_memory and not args.gpu_data
    ds = dmodule.train_dataloader(shuffle=False, pin_memory=use_pinned_memory)
    
    if args.gpu_data:
        ds.dataset.dataset.to("cuda")
    
    build_rom(ds, args.out, tolerance=args.tol, domain=args.domain,
              convergence_on=args.convergence_on,
              use_pinned_memory=use_pinned_memory, 
              prefetch_batches=args.prefetch_batches, 
              freq_cutoff_idx=args.freq_cutoff_idx,
              df=df)
