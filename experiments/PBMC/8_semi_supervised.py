import os
import os.path as osp
from time import perf_counter
from argparse import ArgumentParser
from typing import Optional, Any
from collections import defaultdict

import numpy as np
import pandas as pd
import anndata as ad
import mudata as md
from mmAAVI import MMAAVI
from mmAAVI.preprocess import merge_obs_from_all_modalities
from mmAAVI.utils import setup_seed
from sklearn import metrics as M


def sample_by_batch_label(
    batch: np.ndarray,
    label: np.ndarray,
    use_batch: Any,
    n_per_label: int = 5,
    total_n: Optional[int] = None,
    seed: int = 1,
) -> np.ndarray:
    """
    sample indices from certain batch with all cell types
    """
    batch_ind = (batch == use_batch).nonzero()[0]
    label_batch_used = label[batch_ind]
    label_uni, label_cnt = np.unique(label_batch_used, return_counts=True)
    if (label_cnt < n_per_label).any():
        insu_ind = (label_cnt < n_per_label).nonzero()[0]
        raise ValueError(
            "some label is insufficient: "
            + ",".join(
                "%s %d" % (str(label_uni[i]), label_cnt[i]) for i in insu_ind
            )
        )
    rng = np.random.default_rng(seed)
    res = []
    for li in label_uni:
        res.append(
            rng.choice(
                batch_ind[label_batch_used == li], n_per_label, replace=False
            )
        )
    res = np.concatenate(res)
    if total_n is None or (total_n == res.shape[0]):
        return res

    # 如果total_n不是None，则我们还需要为每个类别补充一些样本
    if total_n < res.shape[0]:
        raise ValueError(
            "total_n can not lower than n_per_label x number of categoricals"
        )
    remain_n = total_n - res.shape[0]
    res_remain = rng.choice(
        np.setdiff1d(batch_ind, res), remain_n, replace=False
    )
    return np.r_[res, res_remain]


def set_semi_supervised_labels(
    mdata: md.MuData,
    nsample: int,
    batch_name: str = "batch",
    label_name: str = "cell_type",
    use_batch: Any = 1,
    nmin_per_seed: str = 5,
    seed: int = 0,
    slabel_name: str = "annotation",
) -> None:
    batch_arr = mdata.obs[batch_name].values
    label_arr = mdata.obs[label_name].values

    slabel = np.full_like(label_arr, fill_value=np.NaN)
    ind = sample_by_batch_label(
        batch_arr,
        label_arr,
        use_batch=use_batch,
        n_per_label=nmin_per_seed,
        seed=seed,
        total_n=nsample,
    )
    slabel[ind] = label_arr[ind]
    mdata.obs[slabel_name] = slabel

    # get label mappings，guarantee the label encoder of train、valid、test
    # is the same
    # categories_all = data.obs[label_name].dropna().unique()
    # categories_l = data.obs["_slabel"].dropna().unique()
    # categories_u = np.setdiff1d(categories_all, categories_l)
    # categories = {
    #     "all": categories_all,
    #     "label": categories_l,
    #     "unlabel": categories_u,
    # }


def evaluate_semi_supervise(
    target_code: np.ndarray,
    proba: np.ndarray,
    batch: Optional[np.ndarray] = None,
) -> pd.DataFrame:
    res = defaultdict(list)

    # ACC
    pred = proba.argmax(axis=1)
    acc = M.accuracy_score(target_code, pred)
    res["ACC"].append(acc)
    # bACC
    bacc = M.balanced_accuracy_score(target_code, pred)
    res["bACC"].append(bacc)
    # recall
    recall = M.recall_score(target_code, pred, average="micro")
    res["recall"].append(recall)
    # precision
    preci = M.precision_score(target_code, pred, average="micro")
    res["precision"].append(preci)
    # AUC
    iden = np.eye(proba.shape[1])
    target_oh = iden[target_code]
    auc = M.roc_auc_score(target_oh, proba, average="micro")
    res["AUC"].append(auc)

    res["scope"].append("global")
    if batch is None:
        return res

    batch_uni = np.unique(batch)
    for bi in batch_uni:
        mask = batch == bi
        target_bi, target_oh_bi, pred_bi, proba_bi = (
            target_code[mask],
            target_oh[mask, :],
            pred[mask],
            proba[mask, :],
        )
        acc = M.accuracy_score(target_bi, pred_bi)
        bacc = M.balanced_accuracy_score(target_bi, pred_bi)
        recall = M.recall_score(target_bi, pred_bi, average="micro")
        preci = M.precision_score(target_bi, pred_bi, average="micro")
        try:
            auc = M.roc_auc_score(target_oh_bi, proba_bi, average="micro")
        except Exception:
            auc = np.NaN

        res["ACC"].append(acc)
        res["bACC"].append(bacc)
        res["recall"].append(recall)
        res["precision"].append(preci)
        res["AUC"].append(auc)
        res["scope"].append(bi)

    res["scope"].append("average")
    for metric in ["ACC", "bACC", "recall", "precision", "AUC"]:
        res[metric].append(np.mean(res[metric][1:]))

    res = pd.DataFrame(res)
    return res


def main():
    parser = ArgumentParser()
    parser.add_argument("--preproc_data_dir", default="./res")
    parser.add_argument("--preproc_data_name", default="pbmc")
    parser.add_argument("--results_dir", default="./res")
    parser.add_argument("--results_name", default="pbmc_semi_sup")
    parser.add_argument("--no_timming", action="store_true")
    parser.add_argument("--seeds", default=list(range(6)), type=int, nargs="+")
    parser.add_argument("--max_epochs", default=300, type=int)
    parser.add_argument("--n_annotations", default=(100,), type=int, nargs="+")
    parser.add_argument("--annotated_batch", default=1, type=int)
    args = parser.parse_args()

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
                ss_label_ratio=0.15
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
