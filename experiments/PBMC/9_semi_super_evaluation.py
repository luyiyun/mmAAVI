import os
import os.path as osp
import re
from argparse import ArgumentParser

import numpy as np
import pandas as pd
import anndata as ad
import mudata as md
import scanpy as sc
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier

from mmAAVI.preprocess import merge_obs_from_all_modalities
from mmAAVI.utils_dev import (
    plot_labeled,
    plot_categories,
    unintegrated_pca,
    evaluate_semi_supervise,
)


def main():
    parser = ArgumentParser()
    parser.add_argument("--preproc_data_dir", default="./res")
    parser.add_argument("--preproc_data_name", default="pbmc")
    parser.add_argument("--results_dir", default="./res")
    parser.add_argument("--semi_results_name", default="pbmc_semi_sup")
    parser.add_argument("--results_name", default="pbmc_semi_sup_eval")
    parser.add_argument("--plot_seed", default=0, type=int)
    parser.add_argument("--plot_n_anno", default=100, type=int)
    parser.add_argument("--add_compar", action="store_true")
    args = parser.parse_args()

    # ========================================================================
    # load preporcessed data and semi-supervised results
    # ========================================================================
    print("========== loading datasets ==========")
    batch_name = "batch"
    label_name = "coarse_cluster"
    mdata_fn = osp.join(
        args.preproc_data_dir, f"{args.preproc_data_name}.h5mu"
    )
    os.makedirs(args.results_dir, exist_ok=True)
    mdata = md.read(mdata_fn)
    merge_obs_from_all_modalities(mdata, key=label_name)
    merge_obs_from_all_modalities(mdata, key=batch_name)

    res_adata = ad.read(
        osp.join(args.results_dir, f"{args.semi_results_name}.h5ad")
    )
    res_csv = pd.read_csv(
        osp.join(args.results_dir, f"{args.semi_results_name}.csv"),
        index_col=0,
    )
    target = mdata.obs[label_name].values
    batch = mdata.obs[batch_name].values

    # ========================================================================
    # calculate the scores of comparison methods
    # ========================================================================
    if args.add_compar:
        print("========== calc scores of comparison methods ==========")
        compar_root_dir = (
            "/mnt/data1/Documents/mosaic-GAN/experiments/PBMC/res/3_comparison"
        )
        compar_methods = [
            "Unintegrated",
            "scmomat",
            "multimap",
            "stabmap",
            "uinmf",
        ]
        compar_methods_name = {
            "Unintegrated": "Unintegrated",
            "scmomat": "scMoMaT",
            "multimap": "MultiMap",
            "uinmf": "UINMF",
            "stabmap": "StabMap",
        }
        # load embedings
        embeds = []
        for methodi in compar_methods:
            if methodi == "Unintegrated":
                embedi = unintegrated_pca(mdata, K=30)
                embeds.append((methodi, embedi, 0))
            else:
                compar_method_dir = osp.join(compar_root_dir, methodi)
                for fn in os.listdir(compar_method_dir):
                    match = re.search(r"pbmc_graph_feats_all_([0-9]).csv", fn)
                    if match:
                        seedi = int(match.group(1))
                        ffn = osp.join(compar_method_dir, fn)
                        embed = pd.read_csv(ffn, index_col=0).values
                        embeds.append((methodi, embed, seedi))

        embeds_compar = {}
        for methodi, embed, seedi in embeds:
            embed_df = pd.DataFrame(embed)
            if embed_df.duplicated().any():
                print("%s-%d has duplicates" % (methodi, seedi))
                mask = embed_df.duplicated().values
                embed[mask, :] = (
                    embed[mask, :]
                    + np.random.randn(mask.sum(), embed.shape[1]) * 1e-3
                )
            key = "%s-%d" % (methodi, seedi)
            embeds_compar[key] = embed

        # mmAAVI_dir = compar_res_mmAAVI[dn]
        # for i in os.listdir(mmAAVI_dir):
        #     keyi = "mmAAVI-%s" % i
        #     mmAAVI_dir_i = osp.join(mmAAVI_dir, i)
        #     embed = torch.load(osp.join(mmAAVI_dir_i, "latents.pt"))
        #     embed = embed.detach().cpu().numpy()
        #     embeds_compar[keyi] = embed

        ss_label = res_adata.obs[f"annotation_{args.plot_n_anno}"]
        unlabel_mask = ss_label.isna().values
        labeled_mask = np.logical_not(unlabel_mask)

        le = LabelEncoder()
        label_enc = le.fit_transform(target)
        label_enc_l = label_enc[labeled_mask]
        label_enc_u = label_enc[unlabel_mask]

        compar_res_df = []
        for k, embed in embeds_compar.items():
            model = KNeighborsClassifier(n_neighbors=10)
            model.fit(embed[labeled_mask], label_enc_l)
            pred_proba = model.predict_proba(embed[unlabel_mask])
            if (
                pred_proba.shape[1] < le.classes_.shape[0]
            ):  # some labels do not exist in training set
                # rename the target
                remap_index = np.zeros(le.classes_.shape[0])
                remap_index[model.classes_] = np.arange(
                    model.classes_.shape[0]
                )
                label_enc_u_remap = remap_index[label_enc_u]
            else:
                label_enc_u_remap = label_enc_u

            res = evaluate_semi_supervise(
                label_enc_u_remap.astype(int),
                pred_proba,
                batch=batch[unlabel_mask],
            )
            methodi, seedi = k.split("-")
            for k, v in zip(
                ["method", "seed"],
                [methodi, seedi],
            ):
                res[k] = v
            compar_res_df.append(res)

        compar_res_df = pd.concat(compar_res_df)
        compar_res_df.replace({"method": compar_methods_name}, inplace=True)

    # ========================================================================
    # calculate the metrics
    # ========================================================================
    print("========== collect the scores of mmAAVI-semi ==========")
    res_csv["method"] = "mmAAVI"
    if args.add_compar:
        res_all = pd.concat([compar_res_df, res_csv])
    else:
        res_all = res_csv
    res_csv_eval = res_all.groupby(["method", "scope"])[
        ["ACC", "bACC", "recall", "precision", "AUC"]
    ].apply(
        lambda df: pd.Series(
            [f"{mu:.4f}±{sig:.4f}" for mu, sig in zip(df.mean(), df.std())],
            index=["ACC", "bACC", "recall", "precision", "AUC"],
        )
    )
    res_csv_eval.to_csv(osp.join(args.results_dir, f"{args.results_name}.csv"))

    # ========================================================================
    # plot umap
    # ========================================================================
    plot_key_obsm = f"mmAAVI-{args.plot_n_anno}_s{args.plot_seed}_z"
    plot_key_pred = f"mmAAVI-{args.plot_n_anno}_s{args.plot_seed}_ss_predict"

    # prepare the dataset used to plotting
    adata_plot_obs = pd.DataFrame(
        {
            "batch": mdata.obs[batch_name].values,
            "target": mdata.obs[label_name].values,
            "pred": res_adata.obs[plot_key_pred].values,
        }
    )
    adata_plot_obs["labeled"] = "label"
    adata_plot_obs.loc[
        res_adata.obs[f"annotation_{args.plot_n_anno}"].isna().values,
        "labeled",
    ] = "no-label"
    adata_plot = ad.AnnData(
        obs=adata_plot_obs, obsm={"mmAAVI": res_adata.obsm[plot_key_obsm]}
    )
    sc.pp.neighbors(
        adata_plot, use_rep="mmAAVI", key_added="mmAAVI", n_neighbors=30
    )
    sc.tl.umap(adata_plot, neighbors_key="mmAAVI", min_dist=0.2)

    # began to plot
    semi_pred = adata_plot.obs
    labeled = semi_pred.labeled.values.copy()
    target = semi_pred.target.values.copy()
    pred = semi_pred.pred.values.copy()
    umap_xy = adata_plot.obsm["X_umap"]

    # target_uni = adata_plot.obs["target"].unique()
    # palette = sns.color_palette(cc.glasbey, n_colors=len(target_uni))
    palette = sns.color_palette()
    fig, axs = plt.subplots(ncols=3, nrows=1, figsize=(16, 4))
    plot_labeled(
        axs[0],
        umap_xy,
        labeled,
        target,
        palette=palette,
        no_label_color="#f2e8cf",
        title="Labeled Seed Cells",
    )
    plot_categories(
        axs[1],
        umap_xy,
        pred,
        palette=palette,
        title="Labeled by mmAAVI-semi",
    )
    plot_categories(
        axs[2],
        umap_xy,
        target,
        palette=palette,
        title="Cell Types",
    )
    fig.tight_layout()

    fig.savefig(osp.join(args.results_dir, f"{args.results_name}.pdf"))
    fig.savefig(
        osp.join(args.results_dir, f"{args.results_name}.png"), dpi=300
    )
    fig.savefig(
        osp.join(args.results_dir, f"{args.results_name}.tiff"), dpi=300
    )


if __name__ == "__main__":
    main()
