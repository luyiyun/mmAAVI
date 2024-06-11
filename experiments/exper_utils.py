from typing import Optional, Sequence, Tuple
from collections import defaultdict

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp
from joblib import Parallel, delayed
from plottable import ColumnDefinition, Table
from plottable.cmap import normed_cmap
from plottable.plots import bar
from sklearn import metrics as M
from sklearn.decomposition import PCA
from tqdm import tqdm


def _compute_clustering_leiden(connectivity_graph, resolution):
    g = sc._utils.get_igraph_from_adjacency(connectivity_graph)
    clustering = g.community_leiden(
        objective_function="modularity",
        weights="weight",
        resolution=resolution,
    )
    clusters = clustering.membership
    return np.asarray(clusters)


def nmi_ari(connectivity_graph, resolution, labels):
    labels_pred = _compute_clustering_leiden(connectivity_graph, resolution)
    nmi = M.normalized_mutual_info_score(labels, labels_pred)
    ari = M.adjusted_rand_score(labels, labels_pred)
    return nmi, ari


def nmi_ari_by_reso(
    adata, keys, resolutions, n_neighbors=15, n_jobs=5, label_name="cell_type"
):
    res = []
    for keyi in tqdm(keys):
        sc.pp.neighbors(
            adata, use_rep=keyi, key_added=keyi, n_neighbors=n_neighbors
        )
        out = Parallel(n_jobs=n_jobs)(
            delayed(nmi_ari)(
                adata.obsp["%s_connectivities" % keyi],
                ri,
                adata.obs[label_name],
            )
            for ri in resolutions
        )
        res.extend(
            [
                {"method": keyi, "resolution": ri, "nmi": nmi_i, "ari": ari_i}
                for (nmi_i, ari_i), ri in zip(out, resolutions)
            ]
        )
    res = pd.DataFrame.from_records(res)
    return res


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
            figsize=(len(res_df.index) * 1.25, 3 + 0.3 * num_embeds)
            if fig_size is None
            else fig_size
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


def save_figure(fg, fn_prefix, dpi=300):
    fg.savefig(fn_prefix + ".pdf")
    fg.savefig(fn_prefix + ".png", dpi=dpi)
    fg.savefig(fn_prefix + ".tiff", dpi=dpi)


def unintegrated_pca(mdata, K: int = 30) -> np.ndarray:
    embeds_pca = []
    for batch in mdata.batch_names:
        Xi = []
        for omic in mdata.omics_names:
            dati = mdata.X[batch, omic]
            if dati is not None:
                if sp.issparse(dati):
                    dati = dati.todense()
                Xi.append(np.asarray(dati))
                # Vi.append(var_dict[omic])
        Xi = np.concatenate(Xi, axis=1)
        embedi = PCA(n_components=K).fit_transform(Xi)
        # Vi = pd.concat(Vi, axis=0)
        # adatai = ad.AnnData(Xi, obs=obs_dict[batch], var=Vi)
        # sc.tl.pca(adatai, n_comps=30, use_highly_variable=False)
        embeds_pca.append(embedi)
    embeds_pca = np.concatenate(embeds_pca, axis=0)

    return embeds_pca
