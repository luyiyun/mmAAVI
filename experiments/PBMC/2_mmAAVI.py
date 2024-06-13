import os
import os.path as osp
from argparse import ArgumentParser

import mudata as md
import scanpy as sc
import anndata as ad
from mmAAVI import MMAAVI
from mmAAVI.preprocess import merge_obs_from_all_modalities
from scib import metrics as scme


def fit_once(
    mdata: md.MuData, save_dir: str, save_name: str, print_metrics: bool = True
) -> None:
    model = MMAAVI(
        dim_c=8,
        input_key="log1p_norm",
        net_key="net",
        balance_sample="max",
        num_workers=4,
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
    parser.add_argument("--results_name", default="pbmc_fit_once")
    args = parser.parse_args()

    mdata_fn = osp.join(
        args.preproc_data_dir, f"{args.preproc_data_name}.h5mu"
    )
    os.makedirs(args.results_dir, exist_ok=True)

    mdata = md.read(mdata_fn)
    merge_obs_from_all_modalities(mdata, key="coarse_cluster")
    merge_obs_from_all_modalities(mdata, key="batch")
    print(mdata)

    fit_once(mdata, args.results_dir, args.results_name)


if __name__ == "__main__":
    main()
