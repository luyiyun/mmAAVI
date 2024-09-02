import os
import os.path as osp
from argparse import ArgumentParser

import mudata as md
import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
from mmAAVI import MMAAVI
from mmAAVI.preprocess import merge_obs_from_all_modalities
from mmAAVI.utils_dev import get_adata_from_mudata, plot_results_table
from mmAAVI.utils import setup_seed
from scib_metrics.benchmark import Benchmarker, BioConservation


def main():
    parser = ArgumentParser()
    parser.add_argument("--preproc_data_dir", default="./res")
    parser.add_argument("--preproc_data_name", default="pbmc")
    parser.add_argument("--results_dir", default="./res")
    parser.add_argument("--results_name", default="pbmc_rm_cell_population")
    parser.add_argument(
        "--remove_batches",
        default=("1",),
        type=str,
        nargs="+",
    )
    parser.add_argument(
        "--remove_cell_type",
        default="Myeloid",
        choices=("Tcell", "NK", "Bcell", "Myeloid"),
        nargs="+",
    )
    parser.add_argument("--num_cluster", default=8, type=int)
    parser.add_argument("--seeds", default=(0,), type=int, nargs="+")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    # ========================================================================
    # load preporcessed data
    # ========================================================================
    batch_name = "batch"
    label_name = "coarse_cluster"

    os.makedirs(args.results_dir, exist_ok=True)

    mdata_fn = osp.join(
        args.preproc_data_dir, f"{args.preproc_data_name}.h5mu"
    )
    save_fn = osp.join(args.results_dir, f"{args.results_name}.h5ad")
    save_fn_csv = osp.join(args.results_dir, f"{args.results_name}.csv")

    mdata = md.read(mdata_fn)
    merge_obs_from_all_modalities(mdata, key=label_name)
    merge_obs_from_all_modalities(mdata, key=batch_name)

    # prepare the container to hold the results
    if not args.overwrite and osp.exists(save_fn):
        res_adata = ad.read(save_fn)
    else:
        res_adata = get_adata_from_mudata(
            mdata,
            obs=[batch_name, label_name],
        )
        res_adata.obs["location"] = np.arange(mdata.n_obs)

    # ========================================================================
    # running for different overlapping portion
    # ========================================================================
    res_csv = {}
    for rm_batches in args.remove_batches:
        # set dataset with different overlapping portion
        rt_ind = ~(
            mdata.obs[batch_name].isin([int(i) for i in rm_batches.split(",")])
            & (mdata.obs[label_name].isin(args.remove_cell_type))
        )
        mdata_r = mdata[rt_ind].copy()
        print(mdata_r)
        for seedi in args.seeds:
            print(
                f"removed cell tyep is {','.join(args.remove_cell_type)}, "
                f"removed batches is {rm_batches}, rand_seed is {seedi}"
            )
            save_name_i = (
                f"rm{','.join(args.remove_cell_type)}_"
                f"rm{rm_batches}_{seedi}"
            )

            setup_seed(seedi)
            model = MMAAVI(
                dim_c=args.num_cluster,
                input_key="log1p_norm",
                net_key="net",
                balance_sample="max",
                num_workers=4,
                seed=seedi,
                deterministic=True,
                device="cuda:1",
            )
            model.fit(mdata_r)

            # record the embeddings
            res_adata.obs[save_name_i] = False
            res_adata.obs.loc[mdata_r.obs.index, save_name_i] = True
            res_adata.obsm[save_name_i] = np.full(
                (res_adata.n_obs, model.dim_z_), np.NaN
            )
            ind_r = res_adata.obs["location"].loc[mdata_r.obs.index]
            res_adata.obsm[save_name_i][ind_r] = mdata_r.obsm["mmAAVI_z"]

            # plot umap
            sc.pp.neighbors(mdata_r, use_rep="mmAAVI_z")
            sc.tl.umap(mdata_r, min_dist=0.2)
            # convert categorical
            mdata_r.obs[batch_name] = (
                mdata_r.obs[batch_name].astype(int).astype("category")
            )
            # plot and save umap
            fig_umap = sc.pl.umap(
                mdata_r,
                color=[batch_name, label_name],
                ncols=2,
                return_fig=True,
            )
            fig_umap.savefig(osp.join(args.results_dir, f"{save_name_i}.png"))

            # calculate the metrics
            adata_plot = get_adata_from_mudata(
                mdata_r, obs=[batch_name, label_name], obsm=["mmAAVI_z"]
            )
            setup_seed(1234)
            bm = Benchmarker(
                adata_plot,
                batch_key=batch_name,
                label_key=label_name,
                embedding_obsm_keys=["mmAAVI_z"],
                n_jobs=8,
                # 其他方法的最优
                bio_conservation_metrics=BioConservation(
                    nmi_ari_cluster_labels_kmeans=False,
                    nmi_ari_cluster_labels_leiden=True,
                ),
            )
            bm.benchmark()
            # save and plot
            res_df = bm.get_results(min_max_scale=False, clean_names=True)
            res_df = res_df.T
            res_csv[save_name_i] = res_df["mmAAVI_z"].astype(float)

    res_csv = pd.DataFrame(res_csv)
    if not args.overwrite:
        res_csv = pd.concat(
            [res_csv, pd.read_csv(save_fn_csv, index_col=0)],
            axis=1,
        )
    else:
        res_csv["Metric Type"] = res_df["Metric Type"].values
    res_csv.to_csv(save_fn_csv)
    plot_results_table(
        res_csv,
        save_name=osp.join(args.results_dir, f"{args.results_name}.svg"),
    )


if __name__ == "__main__":
    main()
