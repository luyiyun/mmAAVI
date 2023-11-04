# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.15.0
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %%
import sys
import os
import os.path as osp
import re
# import pickle
# import warnings
from datetime import datetime

import yaml
from joblib import Parallel, delayed
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
# import colorcet as cc
from sklearn import metrics as M
from tqdm import tqdm
import scanpy as sc
# import anndata as ad
import scipy.sparse as sp
from scib_metrics.benchmark import Benchmarker, BioConservation, BatchCorrection

# %matplotlib inline

# %%
sys.path.append(osp.abspath("../../src"))
from mmAAVI.utils import read_json, setup_seed
from mmAAVI.dataset import MosaicData

# %%
sys.path.append(osp.abspath("../"))
from exper_utils import plot_results_table, save_figure, unintegrated_pca

# %%
root_mmAAVI = "./res/2_mmAAVI/"
res_root = "./res/4_analysis/"
res_root_compar = "./res/4_analysis/compar/"
os.makedirs(res_root, exist_ok=True)
os.makedirs(res_root_compar, exist_ok=True)

# %%
compar_methods = ["Unintegrated", "scmomat", "multimap", "stabmap", "uinmf"]
compar_methods_name = {
    "Unintegrated": "Unintegrated", "scmomat": "scMoMaT", "multimap": "MultiMap", "uinmf": "UINMF", "stabmap": "StabMap"
}

# %%
with open("./res/manual_colors.yaml", "r") as f:
    cell_type_colors = yaml.safe_load(f)

# %%
metric_name_cleaner = {
    "silhouette_label": "Silhouette label",
    "silhouette_batch": "Silhouette batch",
    "isolated_labels": "Isolated labels",
    "nmi_ari_cluster_labels_leiden_nmi": "Leiden NMI",
    "nmi_ari_cluster_labels_leiden_ari": "Leiden ARI",
    "nmi_ari_cluster_labels_kmeans_nmi": "KMeans NMI",
    "nmi_ari_cluster_labels_kmeans_ari": "KMeans ARI",
    "clisi_knn": "cLISI",
    "ilisi_knn": "iLISI",
    "kbet_per_label": "KBET",
    "graph_connectivity": "Graph connectivity",
    "pcr_comparison": "PCR comparison",
}

# %%
setup_seed(1234)

# %% [markdown]
# # Load dataset

# %%
dat = MosaicData.load("./res/1_pp/TripleOmics.mmod")
label_name = "coarse_cell_type"
print(dat)

# %%
adata = dat.to_anndata(sparse=True)
adata.obsm["X_pca"] = unintegrated_pca(dat, K=30)  # 分别进行pca得到
# sc.tl.pca(adata, n_comps=30, use_highly_variable=False)
# 为了后面可以分离anndata
adata.X = sp.csr_matrix(adata.X)
adata.varp["window"] = sp.csr_matrix(adata.varp["window"])
print(adata)

# %% [markdown]
# # Select the dimension of `c`

# %%
# 查找所有的运行结果
runs = []
for di in os.listdir(root_mmAAVI):
    regex = re.search(r"cdimension_graph_feats_([0-9]*?)_(.+?)$", di)
    if regex:
        cdim = int(regex.group(1))
        d = datetime.strptime(regex.group(2), "%Y-%m-%d_%H-%M-%S")
        if d.day >= 29:
            runs.append((cdim, osp.join(root_mmAAVI, di)))
runs

# %%
metric_scores = []
for nc, model_dir_i in runs:
    for seedi in os.listdir(model_dir_i):
        try:
            model_dir_ii = osp.join(model_dir_i, seedi)
            hist_valid = pd.read_csv(osp.join(model_dir_ii, "hist_valid.csv"), index_col=0)
            best = read_json(osp.join(model_dir_ii, "best_score.csv"))
            best_epoch = best["epoch"]
            best_metric = hist_valid.loc[best_epoch, "metric"]
            metric_scores.append({"nc": nc, "seed": seedi, "metric": best_metric})
        except FileNotFoundError:
            pass
metric_scores = pd.DataFrame.from_records(metric_scores)
# metric_scores.head()

# %%
# 计算mean+std
nc_mean_std = metric_scores.groupby("nc")["metric"].apply(lambda x: np.mean(x)+np.std(x)).to_frame().reset_index()
nc_mean_std.columns=["nc", "score"]
# nc_mean_std.head()

# %%
color1, color2 = "tab:blue", "tab:red"

fg = sns.relplot(data=metric_scores, x="nc", y="metric", kind="line", aspect=2, c=color1)
fg.ax.set_xlabel("The number of mixture components")
fg.ax.set_ylabel("The Validation Loss", color=color1)
fg.ax.tick_params(axis='y', labelcolor=color1)

ax2 = fg.ax.twinx()
ax2.plot(nc_mean_std.nc, nc_mean_std.score, color=color2)
ax2.set_ylabel("$\mu+\sigma$", color=color2)
ax2.tick_params(axis='y', labelcolor=color2)

fg.tight_layout()
plt.show()

# %%
fg.savefig(osp.join(res_root, "metric_by_cdimensions.pdf"))
fg.savefig(osp.join(res_root, "metric_by_cdimensions.png"), dpi=300)
fg.savefig(osp.join(res_root, "metric_by_cdimensions.tiff"), dpi=300)

# %% [markdown]
# # Read embeddings obtained by all methods

# %%
# for baseline methods
embeds = []
for methodi in compar_methods:
    if methodi == "Unintegrated":
        embed = adata.obsm["X_pca"]
        embeds.append((methodi, embed, 0))
    else:
        resi = osp.join("./res/3_comparison/", methodi)
        for fn in os.listdir(resi):
            match = re.search(r"TripleOmics_all_([0-9]).csv", fn)
            if match:
                seedi = int(match.group(1))
                ffn = osp.join(resi, fn)
                embed = pd.read_csv(ffn, index_col=0).values
                embeds.append((methodi, embed, seedi))
print(len(embeds))

embed_keys_compar = []
for methodi, embed, seedi in embeds:
    # 计算一下是否存在重复的embeddings
    embed_df = pd.DataFrame(embed)
    if embed_df.duplicated().any():
        print("%s-%d has duplicates" % (methodi, seedi))
        mask = embed_df.duplicated().values
        embed[mask, :] = embed[mask, :] + np.random.randn(mask.sum(), embed.shape[1]) * 1e-3
    key = "%s-%d" % (methodi, seedi)
    adata.obsm[key] = embed
    embed_keys_compar.append(key)
print(embed_keys_compar)

# %%
# for proposed method

# model_dir, nc = "./res/2_mmAAVI/cdimension_graph_feats_10_2023-08-28_20-44-47", 10
model_dir, nc = "./res/2_mmAAVI/cdimension_graph_feats_14_2023-08-29_13-51-22", 14

embed_keys_proposed = []
for i in os.listdir(model_dir):
    model_dir_i = osp.join(model_dir, i)

    if osp.exists(osp.join(model_dir_i, "latents.pt")):
        keyi = "mmAAVI-%s" % i
        embed_keys_proposed.append(keyi)

        embed = torch.load(osp.join(model_dir_i, "latents.pt"))
        embed = embed.detach().cpu().numpy()
        adata.obsm[keyi] = embed

embed_keys_proposed

# %% [markdown] jp-MarkdownHeadingCollapsed=true
# # Evaluate the baseline methods

# %%
# evaluate batch correction and biological conservation based on coarse labels
bm = Benchmarker(
    adata, batch_key="_batch", label_key=label_name, embedding_obsm_keys=embed_keys_compar, n_jobs=8,
    bio_conservation_metrics=BioConservation(nmi_ari_cluster_labels_kmeans=False, nmi_ari_cluster_labels_leiden=True)
)
bm.benchmark()

res_df = bm.get_results(min_max_scale=False, clean_names=True)
res_df = res_df.T
res_df.to_csv(osp.join(res_root_compar, "benchmark_scib.csv"))

# %%
# evaluate biological conservation based on distinct labels for each omics layer
for batch_i, omic_i in zip(dat.batch_names, dat.omics_names):
    adata_i = adata[adata.obs._batch == batch_i].copy()
    bm = Benchmarker(
        adata_i, batch_key="_batch", label_key="cell_type_%s" % omic_i, embedding_obsm_keys=embed_keys_compar, n_jobs=8,
        bio_conservation_metrics=BioConservation(nmi_ari_cluster_labels_kmeans=False, nmi_ari_cluster_labels_leiden=True),
        batch_correction_metrics=BatchCorrection(silhouette_batch=False,
                                                 ilisi_knn=False, kbet_per_label=False,
                                                 graph_connectivity=False, pcr_comparison=False)
    )
    bm.benchmark()
    res_df = bm._results.rename(index=metric_name_cleaner).drop(columns=["Metric Type"])
    res_df.to_csv(osp.join(res_root_compar, "benchmark_scib_%s.csv" % omic_i))

# %% [markdown]
# # Evaluate the proposed method

# %%
# evaluate batch correction and biological conservation based on coarse labels
bm = Benchmarker(
    adata, batch_key="_batch", label_key=label_name, embedding_obsm_keys=embed_keys_proposed, n_jobs=8,
    bio_conservation_metrics=BioConservation(nmi_ari_cluster_labels_kmeans=False, nmi_ari_cluster_labels_leiden=True)
)
bm.benchmark()

res_df = bm.get_results(min_max_scale=False, clean_names=True)
res_df = res_df.T
res_df.to_csv(osp.join(res_root, "benchmark_scib_mmAAVI_nc%d.csv" % nc))

# %%
# evaluate biological conservation based on distinct labels for each omics layer
for batch_i, omic_i in zip(dat.batch_names, dat.omics_names):
    adata_i = adata[adata.obs._batch == batch_i].copy()
    bm = Benchmarker(
        adata_i, batch_key="_batch", label_key="cell_type_%s" % omic_i, embedding_obsm_keys=embed_keys_proposed, n_jobs=8,
        bio_conservation_metrics=BioConservation(nmi_ari_cluster_labels_kmeans=False, nmi_ari_cluster_labels_leiden=True),
        batch_correction_metrics=BatchCorrection(silhouette_batch=False,
                                                 ilisi_knn=False, kbet_per_label=False,
                                                 graph_connectivity=False, pcr_comparison=False)
    )
    bm.benchmark()
    res_df = bm._results.rename(index=metric_name_cleaner).drop(columns=["Metric Type"])
    res_df.to_csv(osp.join(res_root, "benchmark_scib_mmAAVI_nc%d_%s.csv" % (nc, omic_i)))

# %% [markdown]
# # Visualize the evaluation by plotted table

# %%
res_df = []

# batch correction
res_df_batch = pd.read_csv(osp.join(res_root_compar, "benchmark_scib.csv"), index_col=0)
res_df_batch = res_df_batch[res_df_batch["Metric Type"] == "Batch correction"].drop(columns=["Metric Type"])
res_df_batch_mmAAVI = pd.read_csv(osp.join(res_root, "benchmark_scib_mmAAVI_nc%d.csv" % nc), index_col=0)
res_df_batch_mmAAVI = res_df_batch_mmAAVI[res_df_batch_mmAAVI["Metric Type"] == "Batch correction"].drop(columns=["Metric Type"])
res_df_batch = pd.concat([res_df_batch, res_df_batch_mmAAVI], axis=1)
res_df_batch.loc["Batch Correction"] = res_df_batch.mean(axis=0)
res_df_batch["Metric Type"] = ["Batch correction"] * (res_df_batch.shape[0] - 1) + ["Aggregate score"]
res_df.append(res_df_batch)

# bological conservation
for omic_i in ["rna", "atac", "met"]:
    omic_name = {"rna": "RNA", "atac": "ATAC", "met": "Methylation"}[omic_i]
    res_df_omic = pd.read_csv(osp.join(res_root_compar, "benchmark_scib_%s.csv" % omic_i), index_col=0)
    res_df_omic_mmAAVI = pd.read_csv(osp.join(res_root, "benchmark_scib_mmAAVI_nc%d_%s.csv" % (nc, omic_i)), index_col=0)
    res_df_omic = pd.concat([res_df_omic, res_df_omic_mmAAVI], axis=1)
    # res_df_omic.rename(index=lambda x: x+" (%s)" % omic_name, inplace=True)
    res_df_omic.loc["Bio Conservation (%s)" % omic_name] = res_df_omic.mean(axis=0)
    res_df_omic["Metric Type"] = ["Bio Conservation (%s)" % omic_name] * (res_df_batch.shape[0] - 1) + ["Aggregate score"]
    res_df.append(res_df_omic)

res_df = pd.concat(res_df, axis=0)

# total score
temp_df = res_df.drop(columns=["Metric Type"])
total_score = 0.4 * temp_df.loc["Batch Correction", :] + \
    0.2 * (temp_df.loc["Bio Conservation (RNA)", :] + \
           temp_df.loc["Bio Conservation (ATAC)", :] + \
           temp_df.loc["Bio Conservation (Methylation)", :])
res_df.loc["Total", :] = total_score.tolist() + ["Aggregate score"]

# %%
plot_df = res_df.drop(columns=["Metric Type"]).T
method, seed = list(zip(*plot_df.index.map(lambda x:x.split("-")).values))
plot_df["method"] = [compar_methods_name.get(mi, mi) for mi in method]
plot_df["seed"] = seed
plot_df = plot_df.groupby("method").mean().T
plot_df["Metric Type"] = res_df["Metric Type"].values
plot_df = pd.concat([plot_df[plot_df["Metric Type"] != "Aggregate score"],
                     plot_df[plot_df["Metric Type"] == "Aggregate score"]], axis=0)

# %%
# plot 1
plot_results_table(
    plot_df.iloc[list(range(10))+[-5, -4], :],
    row_total="Batch Correction", save_name=osp.join(res_root, "benchmark_scib_batch_1.svg"),
    fig_size=(10*1.5, 3+0.3*6)
)
plt.show()

# %%
# plot 2
plot_results_table(
    plot_df.iloc[list(range(10, 20))+[-3, -2], :],
    row_total="Bio Conservation (ATAC)", save_name=osp.join(res_root, "benchmark_scib_batch_2.svg"),
    fig_size=(10*1.5, 3+0.3*6)
)
plt.show()

# %%
# boxplot for aggregate score
plot_df_box = res_df[res_df["Metric Type"] == "Aggregate score"].drop(columns=["Metric Type"]).T
method, seed = list(zip(*plot_df_box.index.map(lambda x:x.split("-")).values))
plot_df_box["method"] = [compar_methods_name.get(mi, mi) for mi in method]
plot_df_box["seed"] = seed
plot_df_box = plot_df_box.melt(id_vars=["method", "seed"], var_name="metric")
# sns.catplot()
plot_df_box.head()

# %%
plot_df_box_total = plot_df_box.query("metric == 'Total'")
fg = sns.catplot(data=plot_df_box_total, x="value", y="method", col="metric", kind="bar",
                 order=["mmAAVI", "StabMap", "UINMF", "Unintegrated", "scMoMaT", "MultiMap"],
                 aspect=1)
# fg.legend()
fg.set(xlabel="Total", ylabel="")
fg.set_titles("")
fg.despine(top=False, right=False)
save_figure(fg, osp.join(res_root, "total_score"))
plt.show()

# %%
plot_df_box_total.groupby("method").mean().sort_values("value", ascending=False)

# %% [markdown]
# # Visualize UMAP plot

# %% [markdown]
# ## Obtain the UMAP embedding

# %%
embed_keys_umap = ["%s-0" % mn for mn in compar_methods] + ["mmAAVI-4"]
umap_arrs = {}
for k in tqdm(embed_keys_umap):
    sc.pp.neighbors(adata, use_rep=k, key_added=k, n_neighbors=30)
    sc.tl.umap(adata, neighbors_key=k, min_dist=0.2)

    kk = k.split("-")[0]
    kk = compar_methods_name.get(kk, kk)
    umap_arrs[kk] = adata.obsm["X_umap"]

# %%
keys = list(umap_arrs.keys())
keys

# %% [markdown]
# ## Visualize all as one figure contained multiple subfigures

# %%
ks = ["_batch", "cell_type_atac", "cell_type_met", "cell_type_rna"]
used_obs = dat.obs[ks]

# %%
n_methods = len(keys)
batch_colors = sns.color_palette()
batch_uni = np.unique(used_obs["_batch"].values)
markersize_mapper = {"atac": 2.0, "rna": 0.3, "met": 1.4, "batch": 0.2}
# cell_type_uni = np.unique(adata.obs[label_name].values)

fig, axs = plt.subplots(ncols=4, nrows=n_methods, figsize=(4*4, n_methods*3.5))  # , sharex=True, sharey=True)
for i, key in tqdm(enumerate(keys), total=n_methods):

    umap_embed = umap_arrs[key]

    # batch
    ax = axs[i, 0]
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel("")
    ax.set_ylabel(key)
    for j, batch_i in enumerate(batch_uni):
        umap_embed_ij = umap_embed[(used_obs["_batch"] == batch_i).values]
        ax.plot(
            umap_embed_ij[:, 0], umap_embed_ij[:, 1], ".",
            markersize=markersize_mapper["batch"], color=batch_colors[j],
            label=batch_i
        )

    # rna, atac, met
    for n, (bn, omic) in enumerate(zip(dat.batch_names, dat.omics_names)):
        ax = axs[i, n+1]
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel("")
        ax.set_ylabel("")

        mask1 = (used_obs["_batch"] == bn).values
        xy = umap_embed[mask1]
        cell_type_i = used_obs.loc[mask1, "cell_type_%s" % omic].values
        cell_type_i_uni = np.unique(cell_type_i)
        for j, cti in enumerate(cell_type_i_uni):
            xyj = xy[cell_type_i == cti]
            ax.plot(
                xyj[:, 0], xyj[:, 1], ".",
                markersize=markersize_mapper[omic], color=cell_type_colors[cti],
                label=cti
            )

# legend
for i, k in enumerate(["batch"] + list(dat.omics_names)):
    ax = axs[n_methods-1, i]
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(
        handles, labels, loc="upper center", markerscale=10 / markersize_mapper[k], frameon=False, fancybox=False, ncols=3,
        bbox_to_anchor=(0.0, -0.3, 1.0, 0.3), columnspacing=0.2, handletextpad=0.1
    )

# share axis
for i in range(axs.shape[0]):
    axs[i, 0].get_shared_x_axes().join(*axs[i, :])
    axs[i, 0].get_shared_y_axes().join(*axs[i, :])

axs[0, 0].set_title("Batch")
axs[0, 1].set_title("RNA")
axs[0, 2].set_title("ATAC")
axs[0, 3].set_title("Methylation")

plt.show()

# %%
save_figure(fig, osp.join(res_root, "triple-umap-3-%d" % nc))

# %% [markdown]
# ## Visualize distinct figures for proposed method

# %%
k = "mmAAVI-4"
sc.pp.neighbors(adata, use_rep=k, key_added=k, n_neighbors=30)
sc.tl.umap(adata, neighbors_key=k, min_dist=0.2)
umap_embed = adata.obsm["X_umap"]
# umap_embed = umap_arrs["mmAAVI"]

# %%
ks = ["_batch", "cell_type_atac", "cell_type_met", "cell_type_rna"]
used_obs = dat.obs[ks]
batch_colors = sns.color_palette()
batch_uni = np.unique(used_obs["_batch"].values)
markersize_mapper = {"atac": 2.0, "rna": 0.3, "met": 1.4, "batch": 0.2}
# cell_type_uni = np.unique(adata.obs[label_name].values)

# %%
fig, ax = plt.subplots(figsize=(4, 3.5))
# batch
ax.set_xticks([])
ax.set_yticks([])
ax.set_xlabel("")
ax.set_ylabel("")
for j, batch_i in enumerate(batch_uni):
    umap_embed_ij = umap_embed[(used_obs["_batch"] == batch_i).values]
    ax.plot(
        umap_embed_ij[:, 0], umap_embed_ij[:, 1], ".",
        markersize=markersize_mapper["batch"], color=batch_colors[j],
        label=batch_i
    )
ax.set_title("Batch")
handles, labels = ax.get_legend_handles_labels()
ax.legend(
    handles, labels, loc="upper center", markerscale=10/markersize_mapper["batch"],
    frameon=False, fancybox=False, ncols=3,
    bbox_to_anchor=(0.0, -0.3, 1.0, 0.3), columnspacing=0.2, handletextpad=0.1
)
save_figure(fig, osp.join(res_root, "triple-umap--mmAAVI-batch"))
plt.show()
plt.close()

# %%
# rna, atac, met
for n, (bn, omic) in enumerate(zip(dat.batch_names, dat.omics_names)):
    fig, ax = plt.subplots(figsize=(4, 3.5))

    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel("")
    ax.set_ylabel("")

    mask1 = (used_obs["_batch"] == bn).values
    xy = umap_embed[mask1]
    cell_type_i = used_obs.loc[mask1, "cell_type_%s" % omic].values
    cell_type_i_uni = np.unique(cell_type_i)
    for j, cti in enumerate(cell_type_i_uni):
        xyj = xy[cell_type_i == cti]
        ax.plot(
            xyj[:, 0], xyj[:, 1], ".",
            markersize=markersize_mapper[omic], color=cell_type_colors[cti],
            label=cti
        )
    ax.set_title({"batch": "Batch", "rna": "RNA", "met": "Methylation", "atac": "ATAC"}[omic])
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(
        handles, labels, loc="upper center", markerscale=10 / markersize_mapper[omic], frameon=False, fancybox=False, ncols=3,
        bbox_to_anchor=(0.0, -0.3, 1.0, 0.3), columnspacing=0.2, handletextpad=0.1
    )
    save_figure(fig, osp.join(res_root, "triple-umap--mmAAVI-%s" % omic))
    plt.show()
    plt.close()

# %% [markdown]
# # Visualize the repeatation

# %%
df1 = pd.read_csv(osp.join(res_root_compar, "benchmark_scib.csv"), index_col=0).drop(columns=["Metric Type"])
df2 = pd.read_csv(osp.join(res_root, "benchmark_scib_mmAAVI_nc%d.csv" % nc), index_col=0).drop(columns=["Metric Type"])
df = pd.concat([df1, df2], axis=1)

df = df.T
df["method"] = df.index.map(lambda x: x.split("-")[0]).to_series().replace(compar_methods_name).values
df["seed"] = df.index.map(lambda x: int(x.split("-")[1]))

df_long = df.melt(id_vars=["method", "seed"], var_name="metric", value_name="value")

fg = sns.catplot(data=df_long, x="metric", y="value", hue="method", aspect=2.5)
fg.set_xticklabels(rotation=45)
fg.set_xlabels("")
fg.set_ylabels("")
fg.savefig(osp.join(res_root, "scib_duplicates.pdf"))
fg.savefig(osp.join(res_root, "scib_duplicates.png"), dpi=300)
fg.savefig(osp.join(res_root, "scib_duplicates.tiff"), dpi=300)
plt.show()

# %%
