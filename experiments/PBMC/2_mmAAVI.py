import os
import os.path as osp
import logging
from argparse import ArgumentParser

# import torch
# import numpy as np
import mudata as md
import scanpy as sc
import anndata as ad
from mmAAVI import MMAAVI
from mmAAVI.preprocess import merge_obs_from_all_modalities
from scib import metrics as scme

# from scib_metrics.benchmark import Benchmarker


def fit_once(
    mdata: md.MuData, save_dir: str, save_name: str, print_metrics: bool = True
) -> None:
    model = MMAAVI(
        input_key="log1p_norm",
        net_key="net",
        balance_sample="max",
        num_workers=4,
        hiddens_enc_c=(100, 50),
        mix_dec_dot_weight=0.8
    )
    model.fit(mdata)
    mdata.obs["mmAAVI_c_label"] = mdata.obsm["mmAAVI_c"].argmax(axis=1)

    sc.pp.neighbors(mdata, use_rep="mmAAVI_z")
    sc.tl.leiden(mdata, resolution=0.1, key_added="leiden")

    model.differential(mdata, "leiden")

    sc.tl.umap(mdata, min_dist=0.2)
    # convert categorical
    mdata.obs["batch"] = mdata.obs["batch"].astype("category")
    mdata.obs["mmAAVI_c_label"] = mdata.obs["mmAAVI_c_label"].astype(
        "category"
    )
    # plot and save umap
    fig_umap = sc.pl.umap(
        mdata,
        color=["batch", "coarse_cluster", "mmAAVI_c_label", "leiden"],
        ncols=2,
        return_fig=True,
    )
    fig_umap.savefig(osp.join(save_dir, f"{save_name}.png"))
    mdata.write(osp.join(save_dir, f"{save_name}.h5mu"))

    if print_metrics:
        adata = ad.AnnData(obs=mdata.obs[["leiden", "coarse_cluster"]])
        ari = scme.ari(adata, cluster_key="leiden", label_key="coarse_cluster")
        nmi = scme.nmi(adata, cluster_key="leiden", label_key="coarse_cluster")
        print(f"ARI = {ari:.4f}, NMI = {nmi:.4f}")


def main():
    parser = ArgumentParser()
    parser.add_argument("--preproc_data_dir", default="./res")
    parser.add_argument("--preproc_data_name", default="pbmc")
    parser.add_argument("--results_dir", default="./res")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    mdata_fn = osp.join(
        args.preproc_data_dir, f"{args.preproc_data_name}.h5mu"
    )
    os.makedirs(args.results_dir, exist_ok=True)

    mdata = md.read(mdata_fn)
    merge_obs_from_all_modalities(mdata, key="coarse_cluster")
    merge_obs_from_all_modalities(mdata, key="batch")
    print(mdata)
    # batch1_indices = np.nonzero(mdata.obs["batch"] == 1)[0]
    # label_indices = np.random.choice(batch1_indices, 100, replace=False)
    # ss_label = np.full(mdata.n_obs, np.NaN, dtype=object)
    # ss_label[label_indices] = mdata.obs["coarse_cluster"].iloc[label_indices]
    # mdata.obs["semisup_label"] = ss_label

    fit_once(mdata, args.results_dir, "pbmc_fit_once")


if __name__ == "__main__":
    main()