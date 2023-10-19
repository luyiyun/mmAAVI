import os.path as osp
import argparse

import episcanpy as esc


parser = argparse.ArgumentParser()
parser.add_argument("fragment")
parser.add_argument("peak_file")
parser.add_argument("--save_prefix", type=str, default=None)
parser.add_argument("--outdir", type=str, default="./")
parser.add_argument("--peak_size_ext", type=int, default=150)
parser.add_argument("--genome", type=str, default="human")
parser.add_argument("--normalize", action="store_true")
parser.add_argument("--njobs", type=int, default=1)
args = parser.parse_args()

if args.save_prefix is not None:
    save_fn = args.save_prefix + ".h5ad"
else:
    save_fn = osp.splitext(osp.split(args.fragment)[-1])[0] + "_count.h5ad"
save_name = osp.join(args.outdir, save_fn)

# peaks is Dict[chr, list[list[int, int]]]
peaks = esc.ct.load_peaks(args.peak_file)
if args.normalize:
    esc.ct.norm_peaks(peaks, extension=args.peak_size_ext)
esc.ct.bld_mtx_bed(
    fragment_file=args.fragment,
    feature_region=peaks,
    chromosomes=args.genome,
    save=save_name,
    thread=args.njobs
)
