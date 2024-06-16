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
import warnings
from datetime import datetime

from joblib import Parallel, delayed
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import colorcet as cc
from sklearn import metrics as M
from tqdm import tqdm
import scanpy as sc
import anndata as ad
from scib_metrics.benchmark import Benchmarker, BioConservation

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
setup_seed(1234)

# %% [markdown]
# # Load dataset

# %%
dat = MosaicData.load("./res/1_pp/muto2021.mmod")
label_name = "cell_type"
print(dat)

# %%
adata = dat.to_anndata(sparse=True)
adata.obsm["X_pca"] = unintegrated_pca(dat, K=30)  # 分别进行pca得到
# sc.tl.pca(adata, n_comps=30, use_highly_variable=False)
print(adata)

# %% [markdown]
# # Select the dimension of `c`

# %%
# 查找所有的运行结果
time_line = datetime.strptime("2023-09-01_15-00-00", "%Y-%m-%d_%H-%M-%S")
runs = []
for di in os.listdir(root_mmAAVI):
    # repeated_trial_13_2023-09-01_15-07-49
    regex = re.search(r"repeated_trial_(\d+?)_(\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2})$", di)
    if regex:
        cdim = int(regex.group(1))
        d = datetime.strptime(regex.group(2), "%Y-%m-%d_%H-%M-%S")
        if d > time_line:
            runs.append((cdim, osp.join(root_mmAAVI, di)))
runs

# %%
metric_scores = []
for nc, model_dir_i in runs:
    for seedi in os.listdir(model_dir_i):
        model_dir_ii = osp.join(model_dir_i, seedi)
        hist_valid = pd.read_csv(osp.join(model_dir_ii, "hist_valid.csv"), index_col=0)
        best = read_json(osp.join(model_dir_ii, "best_score.csv"))
        best_epoch = best["epoch"]
        best_metric = hist_valid.loc[best_epoch, "metric"]
        metric_scores.append({"nc": nc, "seed": seedi, "metric": best_metric})
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
            match = re.search(r"muto2021_all_([0-9]).csv", fn)
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

model_dir, nc = "./res/2_mmAAVI/repeated_trial_13_2023-09-01_15-07-49/", 13

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
# # %time
bm = Benchmarker(
    adata, batch_key="_batch", label_key=label_name, embedding_obsm_keys=embed_keys_compar, n_jobs=8,
    bio_conservation_metrics=BioConservation(nmi_ari_cluster_labels_kmeans=False, nmi_ari_cluster_labels_leiden=True)
)
bm.benchmark()

# %%
res_df = bm.get_results(min_max_scale=False, clean_names=True)
res_df = res_df.T
res_df.to_csv(osp.join(res_root_compar, "benchmark_scib.csv"))

# %% [markdown] jp-MarkdownHeadingCollapsed=true
# # Evaluate the proposed method

# %%
setup_seed(1234)
bm = Benchmarker(
    adata, batch_key="_batch", label_key=label_name, embedding_obsm_keys=embed_keys_proposed,
    bio_conservation_metrics=BioConservation(
        nmi_ari_cluster_labels_kmeans=False,
        nmi_ari_cluster_labels_leiden=True
    ), n_jobs=5
)
bm.benchmark()

# %%
res_df = bm.get_results(min_max_scale=False)
res_df = res_df.T
res_df.to_csv(osp.join(res_root, "benchmark_scib_mmAAVI_nc%d.csv" % nc))

# %% [markdown] jp-MarkdownHeadingCollapsed=true
# # Visualize the evaluation by plotted table

# %%
res_df = pd.merge(
    pd.read_csv(osp.join(res_root, "benchmark_scib_mmAAVI_nc%d.csv" % nc), index_col=0),
    pd.read_csv(osp.join(res_root_compar, "benchmark_scib.csv"), index_col=0).drop(columns=["Metric Type"]),
    how="outer", left_index=True, right_index=True
)
temp_df = res_df.drop(columns=["Metric Type"]).T
temp_df["method"] = temp_df.index.map(lambda x: x.split("-")[0])
plot_df = temp_df.groupby("method").mean().T
plot_df["Metric Type"] = res_df["Metric Type"]
plot_df.rename(columns=compar_methods_name, inplace=True)
plot_results_table(plot_df, save_name=osp.join(res_root, "scib_results_nc%d.svg" % nc))
plt.show()

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
umap_arrs.keys()

# %% [markdown]
# ## Visualize all as a one figure contained multiple subfigures

# %%
col_names = ["batch5", "omic", "cell_type"]
col_titles = ["Batch", "Omics", "Cell Type"]
col_arrs = [adata.obs[k].values for k in col_names]

# 把omics的label换成大写
col_arrs[1] = np.array([i.upper() for i in col_arrs[1]])
# 把batch换成sample1-5
col_arrs[0] = pd.Series(col_arrs[0]).astype("category").cat.set_categories(["Sample %d" % (i+1) for i in range(5)], rename=True).values

col_uniques = [np.unique(arr) for arr in col_arrs]
col_n_categories = [k.shape[0] for k in col_uniques]
col_palette = [sns.color_palette(), sns.color_palette(), sns.color_palette(cc.glasbey, n_colors=col_n_categories[-1])]


nrow, ncol = len(umap_arrs), len(col_names)
markersize = 15000 / adata.n_obs
markerscale = 10. / markersize

# %%
fig, axs = plt.subplots(ncols=ncol, nrows=nrow, figsize=(ncol*4, nrow*3.5))

for i, (method, xy) in enumerate(umap_arrs.items()):
    for j in range(ncol):
        ax = axs[i, j]
        for c_value, c_color in zip(col_uniques[j], col_palette[j]):
            xyi = xy[col_arrs[j] == c_value, :]
            ax.plot(
                xyi[:, 0], xyi[:, 1], ".",
                markersize=markersize, color=c_color,
                label=c_value if len(c_value) <= 10 else c_value[:10] + ".."  # 有些标签太长，这里只显示其前几位
            )
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel("")
        ax.set_ylabel(method if j == 0 else "")
        if i == 0:
            ax.set_title(col_titles[j])
        if i == (nrow - 1):
            handles, labels = ax.get_legend_handles_labels()
            ax.legend(
                handles, labels, loc="upper center", markerscale=markerscale,
                frameon=False, fancybox=False, ncols=3,
                bbox_to_anchor=(0.0, -0.3, 1.0, 0.3), columnspacing=0.2, handletextpad=0.1
            )

plt.show()

# %%
save_figure(fig, osp.join(res_root, "muto2021-umap"))

# %% [markdown]
# ## Visualize distinct figures for proposed method

# %%
col_names = ["batch5", "omic", "cell_type"]
col_titles = ["Batch", "Omics", "Cell Type"]
col_arrs = [adata.obs[k].values for k in col_names]

# 把omics的label换成大写
col_arrs[1] = np.array([i.upper() for i in col_arrs[1]])
# 把batch换成sample1-5
col_arrs[0] = pd.Series(col_arrs[0]).astype("category").cat.set_categories(["Sample %d" % (i+1) for i in range(5)], rename=True).values

col_uniques = [np.unique(arr) for arr in col_arrs]
col_n_categories = [k.shape[0] for k in col_uniques]
col_palette = [sns.color_palette(), sns.color_palette(), sns.color_palette(cc.glasbey, n_colors=col_n_categories[-1])]

nrow, ncol = len(umap_arrs), len(col_names)
markersize = 15000 / adata.n_obs
markerscale = 10. / markersize

# %%
xy = umap_arrs["mmAAVI"]

for j in range(ncol):
    fig, ax = plt.subplots(figsize=(4, 3.5))
    for c_value, c_color in zip(col_uniques[j], col_palette[j]):
        xyi = xy[col_arrs[j] == c_value, :]
        ax.plot(
            xyi[:, 0], xyi[:, 1], ".",
            markersize=markersize, color=c_color,
            label=c_value if len(c_value) <= 10 else c_value[:10] + ".."  # 有些标签太长，这里只显示其前几位
        )
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_title(col_titles[j])
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(
        handles, labels, loc="upper center", markerscale=markerscale,
        frameon=False, fancybox=False, ncols=3,
        bbox_to_anchor=(0.0, -0.3, 1.0, 0.3), columnspacing=0.2, handletextpad=0.1
    )
    save_figure(fig, osp.join(res_root, "muto2021-umap--mmAAVI-%s" % col_names[j]))
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
