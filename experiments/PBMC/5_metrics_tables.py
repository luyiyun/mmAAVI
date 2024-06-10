import os
import os.path as osp
import re
from argparse import ArgumentParser
from typing import List, Optional, Union, Tuple

import numpy as np
import pandas as pd
import scipy.sparse as sp
import mudata as md
import anndata as ad
import matplotlib
import matplotlib.pyplot as plt
from scib_metrics.benchmark import Benchmarker, BioConservation
from plottable import ColumnDefinition, Table
from plottable.cmap import normed_cmap
from plottable.plots import bar
from mmAAVI.utils import setup_seed


def get_X_from_mudata(
    mdata: md.MuData, sparse: bool = True, fillna: float = 0.0
) -> Union[np.ndarray, sp.csr_matrix]:
    all_X = []
    for m, adat in mdata.mod.items():
        ind_m = mdata.obsm[m]
        Xi = adat.X

        if sparse:
            X_pad = sp.csr_matrix((mdata.n_obs, Xi.shape[1]))
        else:
            X_pad = np.full((mdata.n_obs, Xi.shape[1]), fill_value=fillna)

        if sparse and isinstance(Xi, np.ndarray):
            X_pad[ind_m, :] = sp.csr_matrix(Xi)
        elif sparse and sp.issparse(Xi):
            X_pad[ind_m, :] = Xi
        elif not sparse and isinstance(Xi, np.ndarray):
            X_pad[ind_m, :] = Xi
        elif not sparse and sp.isspmatrix(Xi):
            X_pad[ind_m, :] = Xi.toarray()
        else:
            raise ValueError("X must be a ndarray or sparse matrix.")

        all_X.append(X_pad)

    if sparse:
        return sp.hstack(all_X)
    else:
        return np.concatenate(all_X, axis=1)


def get_adata_from_mudata(
    mdata: md.MuData,
    sparse: bool = True,
    fillna: float = 0.0,
    obs: Optional[List[str]] = None,
    obsm: Optional[List[str]] = None,
) -> ad.AnnData:
    adata = ad.AnnData(X=get_X_from_mudata(mdata, sparse, fillna))
    if obs is not None:
        # if exist categorical column, copy one by one will convert the value
        # of cate column to NaN, but transferring entirely is available.
        adata.obs = mdata.obs[obs]
    if obsm is not None:
        for k in obsm:
            adata.obsm[k] = mdata.obsm[k]

    return adata


def plot_results_table(
    res_df: pd.DataFrame,
    show: bool = True,
    save_name: Optional[str] = None,
    col_metric_type: str = "Metric Type",
    name_aggregate_score: str = "Aggregate score",
    row_total: str = "Total",
    fig_size: Optional[Tuple[float, float]] = None,
) -> Table:
    """Plot the benchmarking results.

    Parameters
    ----------
    min_max_scale
        Whether to min max scale the results.
    show
        Whether to show the plot.
    save_dir
        The directory to save the plot to. If `None`, the plot is not saved.
    """
    # num_embeds = len(self._embedding_obsm_keys)
    cmap_fn = lambda col_data: normed_cmap(
        col_data, cmap=matplotlib.cm.PRGn, num_stds=2.5
    )

    plot_df = (
        res_df.drop(columns=col_metric_type)
        .T.sort_values(by=row_total, ascending=False)
        .astype(np.float64)
    )
    plot_df["Method"] = plot_df.index
    num_embeds = plot_df.shape[0]

    # Split columns by metric type,
    metric_type = res_df[col_metric_type].values
    score_inds = (metric_type == name_aggregate_score).nonzero()[0]
    other_inds = (metric_type != name_aggregate_score).nonzero()[0]
    score_titles = plot_df.columns.values[score_inds]
    other_titles = plot_df.columns.values[other_inds]
    # create unique names as index, origin names as the column title
    plot_df.columns = [
        "%s_%d" % (ind, i)
        for i, ind in enumerate(plot_df.columns[:-1])
        # expect Method
    ] + ["Method"]
    score_names = plot_df.columns.values[score_inds]
    other_names = plot_df.columns.values[other_inds]

    column_definitions = [
        ColumnDefinition(
            "Method", width=1.5, textprops={"ha": "left", "weight": "bold"}
        ),
    ]
    # Circles for the metric values
    column_definitions += [
        ColumnDefinition(
            col,
            title=titlei.replace(" ", "\n", 2),
            width=1,
            textprops={
                "ha": "center",
                "bbox": {"boxstyle": "circle", "pad": 0.25},
            },
            cmap=cmap_fn(plot_df.iloc[:, i]),
            group=metric_type[i],
            formatter="{:.2f}",
        )
        for i, col, titlei in zip(other_inds, other_names, other_titles)
    ]
    # Bars for the aggregate scores
    column_definitions += [
        ColumnDefinition(
            col,
            width=1,
            title=titlei.replace(" ", "\n", 2),
            plot_fn=bar,
            plot_kw={
                "cmap": matplotlib.cm.YlGnBu,
                "plot_bg_bar": False,
                "annotate": True,
                "height": 0.9,
                "formatter": "{:.2f}",
            },
            group=metric_type[i],
            border="left" if ind == 0 else None,
        )
        for ind, (i, col, titlei) in enumerate(
            zip(score_inds, score_names, score_titles)
        )
    ]
    # Allow to manipulate text post-hoc (in illustrator)
    with matplotlib.rc_context({"svg.fonttype": "none"}):
        fig, ax = plt.subplots(
            figsize=(
                (len(res_df.index) * 1.25, 3 + 0.3 * num_embeds)
                if fig_size is None
                else fig_size
            )
        )
        tab = Table(
            plot_df,
            cell_kw={
                "linewidth": 0,
                "edgecolor": "k",
            },
            column_definitions=column_definitions,
            ax=ax,
            row_dividers=True,
            footer_divider=True,
            textprops={"fontsize": 10, "ha": "center"},
            row_divider_kw={"linewidth": 1, "linestyle": (0, (1, 5))},
            col_label_divider_kw={"linewidth": 1, "linestyle": "-"},
            column_border_kw={"linewidth": 1, "linestyle": "-"},
            index_col="Method",
        ).autoset_fontcolors(colnames=plot_df.columns)
    if show:
        plt.show()
    if save_name is not None:
        fig.savefig(save_name, facecolor=ax.get_facecolor(), dpi=300)

    return tab, fig


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
