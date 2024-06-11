import os
import os.path as osp
import re
from argparse import ArgumentParser

import mudata as md
import anndata as ad
from scib_metrics.benchmark import Benchmarker, BioConservation
from mmAAVI.utils import setup_seed
from mmAAVI.utils_dev import get_adata_from_mudata, plot_results_table


def main():
    parser = ArgumentParser()
    parser.add_argument("--results_dir", default="./res")
    parser.add_argument("--mmaavi_result", default="pbmc_decide_num_clusters")
    parser.add_argument("--compar_result", default="pbmc_comparison")
    parser.add_argument("--save_csv_name", default="pbmc_metrics")
    parser.add_argument("--save_fig_name", default="pbmc_metrics")
    args = parser.parse_args()

    # ========================================================================
    # load trained results
    # ========================================================================
    os.makedirs(args.results_dir, exist_ok=True)
    mmaavi_result_fn = osp.join(args.results_dir, f"{args.mmaavi_result}.h5mu")
    mdata_mmaavi = md.read(mmaavi_result_fn)
    adata_compar = ad.read(
        osp.join(args.results_dir, f"{args.compar_result}.h5ad")
    )

    # ========================================================================
    # create anndata for benchmarking
    # ========================================================================
    batch_name = "batch__code"
    label_name = "coarse_cluster"

    mdata = mdata_mmaavi
    nc_best = mdata.uns["best_nc"]
    pattern = rf"mmAAVI_nc{nc_best}_s(\d+?)_z"
    random_seeds, embed_keys = [], []
    for k in mdata.obsm.keys():
        res = re.search(pattern, k)
        if res:
            random_seeds.append(int(res.group(1)))
            embed_keys.append(k)

    # create anndata
    adata = get_adata_from_mudata(
        mdata,
        obs=[batch_name, label_name],
        obsm=embed_keys,
    )

    # load results of comparison methods
    for k in adata_compar.obsm.keys():
        adata.obsm[k] = adata_compar.obsm[k]
        embed_keys.append(k)

    # ========================================================================
    # benchmarking
    # ========================================================================
    setup_seed(1234)
    bm = Benchmarker(
        adata,
        batch_key=batch_name,
        label_key=label_name,
        embedding_obsm_keys=embed_keys,
        n_jobs=8,
        # 其他方法的最优
        bio_conservation_metrics=BioConservation(
            nmi_ari_cluster_labels_kmeans=False,
            nmi_ari_cluster_labels_leiden=True,
        ),
    )
    bm.benchmark()

    # ========================================================================
    # save and plot
    # ========================================================================
    res_df = bm.get_results(min_max_scale=False, clean_names=True)
    res_df = res_df.T
    res_df[embed_keys] = res_df[embed_keys].astype(float)
    res_df.to_csv(osp.join(args.results_dir, f"{args.save_csv_name}.csv"))

    metric_type = res_df["Metric Type"].values
    temp_df = res_df.drop(columns=["Metric Type"]).T
    temp_df["method"] = temp_df.index.map(lambda x: x.split("_")[0])
    plot_df = temp_df.groupby("method").mean().T
    plot_df["Metric Type"] = metric_type
    # plot_df.rename(columns=compar_methods_name, inplace=True)
    plot_results_table(
        plot_df,
        save_name=osp.join(args.results_dir, f"{args.save_fig_name}.svg"),
    )


if __name__ == "__main__":
    main()
