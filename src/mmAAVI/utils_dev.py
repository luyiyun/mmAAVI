from collections import defaultdict
from typing import List, Optional, Union, Tuple, Any

import numpy as np
import pandas as pd
import scipy.sparse as sp
import mudata as md
import anndata as ad
import matplotlib
import matplotlib.pyplot as plt
import sklearn.metrics as M
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from plottable import ColumnDefinition, Table
from plottable.cmap import normed_cmap
from plottable.plots import bar


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


def sample_by_batch_label(
    batch: np.ndarray,
    label: np.ndarray,
    use_batch: Optional[Any] = None,
    n_per_label: int = 5,
    total_n: Optional[int] = None,
    seed: int = 1,
) -> np.ndarray:
    """
    sample indices from certain batch with all cell types
    """
    if use_batch is None:
        label_batch_used = label
        batch_ind = np.arange(len(label))
    else:
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
    # res_remain = rng.choice(
    #     np.setdiff1d(batch_ind, res), remain_n, replace=False,
    # )
    selected_indice = np.setdiff1d(batch_ind, res)
    # select stratified by label
    res_remain, _ = train_test_split(
        selected_indice,
        train_size=remain_n,
        shuffle=True,
        random_state=seed,
        stratify=label[selected_indice],
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


def unintegrated_pca(
    mdata: md.MuData, batch_name: str = "batch", K: int = 30
) -> np.ndarray:
    batch_arr = mdata.obs[batch_name].values
    batch_uni = np.unique(batch_arr)

    embeds = np.zeros((mdata.n_obs, K))
    for bi in batch_uni:
        mask = batch_arr == bi
        mdatai = mdata[mask, :]
        X_bi = []
        for adatai in mdatai.mod.values():
            if adatai.n_obs == 0:
                continue
            Xi = adatai.X
            if sp.issparse(Xi):
                Xi = Xi.todense()
            X_bi.append(np.asarray(Xi))
        X_bi = np.concatenate(X_bi, axis=1)
        embedi = PCA(n_components=K).fit_transform(X_bi)
        embeds[mask, :] = embedi

    return embeds
