import os
import os.path as osp
import logging
from time import perf_counter
from argparse import ArgumentParser

import numpy as np
import pandas as pd
import mudata as md
from mmAAVI import MMAAVI
from mmAAVI.preprocess import merge_obs_from_all_modalities
import seaborn as sns


def main():
    parser = ArgumentParser()
    parser.add_argument("--preproc_data_dir", default="./res")
    parser.add_argument("--preproc_data_name", default="pbmc")
    parser.add_argument("--results_dir", default="./res")
    parser.add_argument("--results_name", default="pbmc_decide_num_clusters")
    parser.add_argument("--no_timming", action="store_true")
    parser.add_argument(
        "--num_clusters", default=list(range(3, 11)), type=int, nargs="+"
    )
    parser.add_argument("--seeds", default=list(range(6)), type=int, nargs="+")
    parser.add_argument("--max_epochs", default=300, type=int)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    # ========================================================================
    # load preporcessed data
    # ========================================================================
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

    # ========================================================================
    # running for different dimension of c
    # ========================================================================
    timing = not args.no_timming
    valid_metrics = []
    if timing:
        res_timing = []
    for nc in args.num_clusters:  # range(3, 11):
        print("nc = %d" % nc)

        for seedi in args.seeds:
            print(f"num_cluster = {nc}, rand_seed = {seedi}")
            key_add = f"mmAAVI_nc{nc}_s{seedi}"
            # set random seed
            # deterministic保证重复性，但是性能慢两倍
            # setup_seed(seedi, deterministic=True)
            # set path which contains model and results
            # save_dir_i = osp.join(save_dir, str(seedi))
            # os.makedirs(save_dir_i, exist_ok=True)
            # print(save_dir)

            model = MMAAVI(
                dim_c=nc,
                input_key="log1p_norm",
                net_key="net",
                # balance_sample="max",
                num_workers=4,
                hiddens_enc_c=(100, 50),
                seed=seedi,
                deterministic=True,
                max_epochs=args.max_epochs,
            )
            if timing:
                t1 = perf_counter()
            model.fit(mdata, key_add=key_add)
            if timing:
                res_timing.append((nc, seedi, perf_counter() - t1))

            # collect valid metrics
            best_epoch = model.train_best_["epoch"]
            best_metric = model.train_hists_["valid"].loc[best_epoch, "metric"]
            valid_metrics.append((nc, seedi, best_metric))

    if timing:
        mdata.uns["timing"] = res_timing
    mdata.uns["valid_metrics"] = valid_metrics

    # ========================================================================
    # decide the best number of clusters
    # ========================================================================
    df_metrics = pd.DataFrame.from_records(
        valid_metrics, columns=["nc", "seed", "metric"]
    )
    nc_mean_std = (
        df_metrics.groupby("nc")["metric"]
        .apply(lambda x: np.mean(x) + np.std(x))
        .to_frame()
        .reset_index()
    )
    nc_mean_std.columns = ["nc", "score"]
    best_nc = nc_mean_std.sort_values("score")["nc"].iloc[0]
    mdata.uns["best_nc"] = best_nc

    # ========================================================================
    # save the training results
    # ========================================================================
    mdata.write(osp.join(args.results_dir, f"{args.results_name}.h5mu"))

    # ========================================================================
    # plot the valid metric loss
    # ========================================================================
    color1, color2 = "tab:blue", "tab:red"
    fg = sns.relplot(
        data=df_metrics, x="nc", y="metric", kind="line", aspect=2, c=color1
    )
    fg.ax.set_xlabel("The number of mixture components")
    fg.ax.set_ylabel("The Validation Loss", color=color1)
    fg.ax.tick_params(axis="y", labelcolor=color1)

    ax2 = fg.ax.twinx()
    ax2.plot(nc_mean_std.nc, nc_mean_std.score, color=color2)
    ax2.set_ylabel(r"$\mu+\sigma$", color=color2)
    ax2.tick_params(axis="y", labelcolor=color2)
    fg.tight_layout()
    fg.savefig(osp.join(args.results_dir, f"{args.results_name}.pdf"))
    fg.savefig(
        osp.join(args.results_dir, f"{args.results_name}.png"), dpi=300
    )
    fg.savefig(
        osp.join(args.results_dir, f"{args.results_name}.tiff"), dpi=300
    )


if __name__ == "__main__":
    main()

# # %%
# # running for subsampling experiments
# # data_fns = [
# #     osp.join(data_dir, fn)
# #     for fn in os.listdir(data_dir)
# #     if re.search(r"pbmc_[0-9]*?_[0-9].mmod", fn)
# # ]
# # for i, data_fni in enumerate(data_fns):
# #     # res_fn = osp.join(res_dir, "%s.csv" % osp.basename(data_fni)[:-5])
# #     prefix = "subample_%s_" % ("_".join(data_fni[:-5].split("_")[-2:]))
# #     print("%d/%d %s" % (i+1, len(data_fns), prefix))
# #     run(
# #         dat=data_fni,
# #         save_root=res_dir,
# #         trial_prefix=prefix,
# #         seeds=[0]
# #     )
