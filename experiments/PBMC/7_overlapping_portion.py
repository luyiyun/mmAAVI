import os
import os.path as osp
import logging
from time import perf_counter
from argparse import ArgumentParser

import numpy as np
import anndata as ad
import mudata as md
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
    parser.add_argument("--results_name", default="pbmc_overlap_portion")
    parser.add_argument("--no_timming", action="store_true")
    parser.add_argument(
        "--remove_omic", default="protein", choices=("atac", "rna", "protein")
    )
    parser.add_argument(
        "--remove_batches",
        default=("1", "1,2", "1,2,3", "1,2,3,4"),
        type=str,
        nargs="+",
    )
    parser.add_argument("--num_cluster", default=8, type=int)
    parser.add_argument("--seeds", default=list(range(6)), type=int, nargs="+")
    parser.add_argument("--max_epochs", default=300, type=int)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    # ========================================================================
    # load preporcessed data
    # ========================================================================
    batch_name = "batch"
    label_name = "coarse_cluster"
    mdata_fn = osp.join(
        args.preproc_data_dir, f"{args.preproc_data_name}.h5mu"
    )
    os.makedirs(args.results_dir, exist_ok=True)
    mdata = md.read(mdata_fn)
    merge_obs_from_all_modalities(mdata, key=label_name)
    merge_obs_from_all_modalities(mdata, key=batch_name)

    # prepare the container to hold the results
    res_adata = ad.AnnData(obs={"placeholder": np.arange(mdata.n_obs)})

    # batch counts
    batch_counts = mdata.obs["batch"].astype(int).value_counts().to_dict()

    # ========================================================================
    # running for different overlapping portion
    # ========================================================================
    timing = not args.no_timming

    valid_metrics = []
    if timing:
        res_timing = []
    for rm_batches in args.remove_batches:

        # set dataset with different overlapping portion
        mdata_rb = mdata.copy()
        rm_omic = mdata_rb.mod[args.remove_omic]
        ind = ~rm_omic.obs["batch"].isin(
            [int(i) for i in rm_batches.split(",")]
        )
        if ind.any():
            mdata_rb = md.MuData(
                data={
                    k: v.copy() if k != args.remove_omic else v[ind].copy()
                    for k, v in mdata_rb.mod.items()
                },
                obs=mdata_rb.obs[[batch_name, label_name]].copy(),
                var=mdata_rb.var.copy(),
                varp=mdata_rb.varp.copy()
            )
        else:  # if remove all samples, then remove the whole omic data
            remain_var_ind = np.bitwise_not(mdata_rb.varm[args.remove_omic])
            mdata_rb = md.MuData(
                data={
                    k: v.copy()
                    for k, v in mdata_rb.mod.items()
                    if k != args.remove_omic
                },
                obs=mdata_rb.obs[[batch_name, label_name]].copy(),
                var=mdata_rb.var.loc[remain_var_ind, :].copy(),
                varp={
                    k: v[remain_var_ind, :][:, remain_var_ind]
                    for k, v in mdata_rb.varp.items()
                },
            )
        print(mdata_rb)

        for seedi in args.seeds:
            print(
                f"removed omics is {args.remove_omic}, "
                f"removed batches is {rm_batches}, rand_seed is {seedi}"
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
                max_epochs=args.max_epochs,
                device="cuda:1"
            )
            if timing:
                t1 = perf_counter()
            model.fit(mdata_rb)
            if timing:
                res_timing.append((rm_batches, seedi, perf_counter() - t1))

            # collect the embeddings
            rm_batches_int = [int(i) for i in rm_batches.split(",")]
            ratio = (
                sum(
                    v
                    for k, v in batch_counts.items()
                    if k not in rm_batches_int
                )
                / mdata.n_obs
            ) * 100
            for postfix in ["z", "att", "c"]:
                res_adata.obsm[f"mmAAVI-{ratio:.0f}%_s{seedi}_{postfix}"] = (
                    mdata_rb.obsm[f"mmAAVI_{postfix}"].copy()
                )

            # collect valid metrics
            best_epoch = model.train_best_["epoch"]
            best_metric = model.train_hists_["valid"].loc[best_epoch, "metric"]
            valid_metrics.append((rm_batches, seedi, best_metric))

    if timing:
        res_adata.uns["timing"] = res_timing
    res_adata.uns["valid_metrics"] = valid_metrics

    # ========================================================================
    # save the results
    # ========================================================================
    res_adata.write(osp.join(args.results_dir, f"{args.results_name}.h5ad"))

    # ========================================================================
    # plot the metric table
    # ========================================================================
    # create anndata
    adata_plot = get_adata_from_mudata(
        mdata,
        obs=[batch_name, label_name],
    )
    # load results from res_adata
    for k in res_adata.obsm.keys():
        if k.endswith("_z"):
            adata_plot.obsm[k] = res_adata.obsm[k]
    # benchmarking
    embed_keys = list(adata_plot.obsm.keys())
    setup_seed(1234)
    bm = Benchmarker(
        adata_plot,
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
    # save and plot
    res_df = bm.get_results(min_max_scale=False, clean_names=True)
    res_df = res_df.T
    res_df[embed_keys] = res_df[embed_keys].astype(float)
    res_df.to_csv(osp.join(args.results_dir, f"{args.results_name}.csv"))
    metric_type = res_df["Metric Type"].values
    temp_df = res_df.drop(columns=["Metric Type"]).T
    temp_df["method"] = temp_df.index.map(lambda x: x.split("_")[0])
    plot_df = temp_df.groupby("method").mean().T
    plot_df["Metric Type"] = metric_type
    # plot_df.rename(columns=compar_methods_name, inplace=True)
    plot_results_table(
        plot_df,
        save_name=osp.join(args.results_dir, f"{args.results_name}.svg"),
    )


if __name__ == "__main__":
    main()
