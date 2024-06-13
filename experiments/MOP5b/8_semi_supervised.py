import os
import os.path as osp
from time import perf_counter
from argparse import ArgumentParser

import numpy as np
import pandas as pd
import anndata as ad
import mudata as md
from mmAAVI import MMAAVI
from mmAAVI.preprocess import merge_obs_from_all_modalities
from mmAAVI.utils import setup_seed
from mmAAVI.utils_dev import (
    set_semi_supervised_labels,
    evaluate_semi_supervise,
)


def main():
    parser = ArgumentParser()
    parser.add_argument("--preproc_data_dir", default="./res")
    parser.add_argument("--preproc_data_name", default="mop5b")
    parser.add_argument("--results_dir", default="./res")
    parser.add_argument("--results_name", default="mop5b_semi_sup")
    parser.add_argument("--no_timming", action="store_true")
    parser.add_argument("--seeds", default=list(range(6)), type=int, nargs="+")
    parser.add_argument("--max_epochs", default=300, type=int)
    parser.add_argument("--n_annotations", default=(100,), type=int, nargs="+")
    # default select annotations from all batches
    parser.add_argument("--annotated_batch", default=None, type=int)
    args = parser.parse_args()

    # ========================================================================
    # load preporcessed data
    # ========================================================================
    batch_name = "batch"
    label_name = "cell_type"
    mdata_fn = osp.join(
        args.preproc_data_dir, f"{args.preproc_data_name}.h5mu"
    )
    os.makedirs(args.results_dir, exist_ok=True)
    mdata = md.read(mdata_fn)
    merge_obs_from_all_modalities(mdata, key=label_name)
    merge_obs_from_all_modalities(mdata, key=batch_name)

    # prepare the container to hold the results
    res_adata = ad.AnnData(obs={"placeholder": np.arange(mdata.n_obs)})

    # ========================================================================
    # run semi-supervised learning
    # ========================================================================
    timing = not args.no_timming
    valid_metrics = []
    if timing:
        res_timing = []
    eval_df = []
    for n_anno_i in args.n_annotations:

        # prepare the semi-supervised labels
        slabel_name = f"annotation_{n_anno_i}"
        set_semi_supervised_labels(
            mdata,
            nsample=n_anno_i,
            batch_name=batch_name,
            label_name=label_name,
            use_batch=args.annotated_batch,
            seed=0,
            slabel_name=slabel_name,
            nmin_per_seed=2,
        )
        res_adata.obs[slabel_name] = mdata.obs[slabel_name].values
        unlabel_ind = res_adata.obs[slabel_name].isna().values
        label_ss = mdata.obs[label_name].values[unlabel_ind]

        for seedi in args.seeds:
            print(f"number of annotations is {n_anno_i}, rand_seed is {seedi}")

            # set random seed
            # deterministic保证重复性，但是性能慢两倍
            setup_seed(seedi, deterministic=True)

            # TODO: load unsupervsed pretrained model??
            model = MMAAVI(
                input_key="log1p_norm",
                net_key="net",
                balance_sample="max",
                num_workers=4,
                seed=seedi,
                deterministic=True,
                max_epochs=args.max_epochs,
                device="cuda:1",
                sslabel_key=slabel_name,
                ss_label_ratio=0.2,
            )
            if timing:
                t1 = perf_counter()
            model.fit(mdata)
            if timing:
                res_timing.append((n_anno_i, seedi, perf_counter() - t1))

            # collect the embeddings
            for postfix in ["z", "att", "c"]:
                res_adata.obsm[f"mmAAVI-{n_anno_i}_s{seedi}_{postfix}"] = (
                    mdata.obsm[f"mmAAVI_{postfix}"].copy()
                )
            # collect the ss prediction
            res_adata.obs[f"mmAAVI-{n_anno_i}_s{seedi}_ss_predict"] = (
                mdata.obs["mmAAVI_ss_predict"].values
            )

            # collect valid metrics
            best_epoch = model.train_best_["epoch"]
            best_metric = model.train_hists_["valid"].loc[best_epoch, "metric"]
            valid_metrics.append((n_anno_i, seedi, best_metric))

            # calculate the metric scores
            proba = mdata.obsm["mmAAVI_c"]
            proba_ss = proba[unlabel_ind, :]
            label_ss_code = model.sslabel_enc_.transform(label_ss)
            eval_df_i = evaluate_semi_supervise(
                label_ss_code,
                proba_ss,
                mdata.obs[batch_name].values[unlabel_ind],
            )
            eval_df_i["seed"] = seedi
            eval_df_i["n_anno"] = n_anno_i
            eval_df.append(eval_df_i)

    if timing:
        res_adata.uns["timing"] = res_timing
    res_adata.uns["valid_metrics"] = valid_metrics
    eval_df = pd.concat(eval_df)

    # ========================================================================
    # save the results
    # ========================================================================
    res_adata.write(osp.join(args.results_dir, f"{args.results_name}.h5ad"))
    eval_df.to_csv(osp.join(args.results_dir, f"{args.results_name}.csv"))


if __name__ == "__main__":
    main()
