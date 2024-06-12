import os
import os.path as osp
from argparse import ArgumentParser

import numpy as np
import pandas as pd
import anndata as ad
import mudata as md
import scanpy as sc
import matplotlib.pyplot as plt
import seaborn as sns

# import colorcet as cc
from mmAAVI.preprocess import merge_obs_from_all_modalities


def plot_labeled(
    ax,
    umap_xy,
    label,
    target,
    palette,
    label_makersize=5.0,
    leg_ncols=1,
    no_label_color="gray",
    title=None,
):
    leg_loc = "best"
    bbox_to_anchor = (1.0, 0.0, 0.3, 1.0)

    categories = np.unique(target[label == "label"]).tolist()
    for i, labeli in enumerate(["no-label"] + categories):
        if labeli == "no-label":
            xyi = umap_xy[label == labeli, :]
        else:
            xyi = umap_xy[(label == "label") & (target == labeli), :]
        if labeli == "no-label":
            colori = no_label_color
        elif isinstance(palette, dict):
            colori = palette[labeli]
        else:
            colori = palette[i - 1]
        ax.plot(
            xyi[:, 0],
            xyi[:, 1],
            "." if labeli == "no-label" else "*",
            label=labeli,
            markersize=(
                10000 / umap_xy.shape[0]
                if labeli == "no-label"
                else label_makersize
            ),
            color=colori,
        )
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel("")
    ax.set_ylabel("")
    if title is not None:
        ax.set_title(title)

    handles, labels = ax.get_legend_handles_labels()
    leg = ax.legend(
        handles,
        labels,
        loc=leg_loc,
        frameon=False,
        fancybox=False,
        ncols=leg_ncols,
        bbox_to_anchor=bbox_to_anchor,
        columnspacing=0.2,
        handletextpad=0.1,
    )

    for h in leg.legend_handles:
        h.set_markersize(10.0)
    leg.set_in_layout(True)

    return ax, leg


# plot scatter plot for a specific categorical variable
def plot_categories(ax, umap_xy, feature, palette, leg_ncols=1, title=None):
    leg_loc = "best"
    bbox_to_anchor = (1.0, 0.0, 0.3, 1.0)

    categories = np.unique(feature).tolist()
    for i, labeli in enumerate(categories):
        xyi = umap_xy[feature == labeli, :]
        ax.plot(
            xyi[:, 0],
            xyi[:, 1],
            ".",
            label=labeli,
            markersize=10000 / umap_xy.shape[0],
            color=palette[labeli] if isinstance(palette, dict) else palette[i],
        )
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel("")
    ax.set_ylabel("")
    if title is not None:
        ax.set_title(title)

    handles, labels = ax.get_legend_handles_labels()
    leg = ax.legend(
        handles,
        labels,
        loc=leg_loc,
        frameon=False,
        fancybox=False,
        ncols=leg_ncols,
        bbox_to_anchor=bbox_to_anchor,
        columnspacing=0.2,
        handletextpad=0.1,
    )

    for h in leg.legend_handles:
        h.set_markersize(10.0)
    leg.set_in_layout(True)

    return ax, leg


def main():
    parser = ArgumentParser()
    parser.add_argument("--preproc_data_dir", default="./res")
    parser.add_argument("--preproc_data_name", default="pbmc")
    parser.add_argument("--results_dir", default="./res")
    parser.add_argument("--semi_results_name", default="pbmc_semi_sup")
    parser.add_argument("--results_name", default="pbmc_semi_sup_eval")
    parser.add_argument("--plot_seed", default=0, type=int)
    parser.add_argument("--plot_n_anno", default=100, type=int)
    args = parser.parse_args()

    # ========================================================================
    # load preporcessed data and semi-supervised results
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

    res_adata = ad.read(
        osp.join(args.results_dir, f"{args.semi_results_name}.h5ad")
    )
    res_csv = pd.read_csv(
        osp.join(args.results_dir, f"{args.semi_results_name}.csv"),
        index_col=0,
    )

    # ========================================================================
    # calculate the metrics
    # ========================================================================
    res_csv_eval = res_csv.groupby("scope")[
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
