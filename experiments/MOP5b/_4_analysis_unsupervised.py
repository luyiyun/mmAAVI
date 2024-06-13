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
import scipy.sparse as sp

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
# 工具函数，保存matplotlib figure
def save_figure(fg, fn_prefix, dpi=300):
    fg.savefig(fn_prefix+".pdf")
    fg.savefig(fn_prefix+".png", dpi=dpi)
    fg.savefig(fn_prefix+".tiff", dpi=dpi)


# %% [markdown]
# # 读取数据

# %%
dat = MosaicData.load("./res/1_pp/mop5b_full.mmod")
label_name = "cell_type"
# fine_label = "cluster"
print(dat)

# %%
adata = dat.to_anndata(sparse=True)
adata.obsm["X_pca"] = unintegrated_pca(dat, K=30)
adata.X = sp.csr_matrix(adata.X)
adata.varp["window"] = sp.csr_matrix(adata.varp["window"])
print(adata)

# %% [markdown]
# # 评价对比方法

# %% [markdown]
# ## 载入embedding

# %%
embeds = []
for methodi in compar_methods:
    if methodi == "Unintegrated":
        embed = adata.obsm["X_pca"]
        embeds.append((methodi, embed, 0))
    else:
        resi = osp.join("./res/3_comparison/", methodi)
        for fn in os.listdir(resi):
            match = re.search(r"mop5b_full_all_([0-9]).csv", fn)
            if match:
                seedi = int(match.group(1))
                ffn = osp.join(resi, fn)
                embed = pd.read_csv(ffn, index_col=0).values
                embeds.append((methodi, embed, seedi))
print(len(embeds))

embed_keys = []
for methodi, embed, seedi in embeds:
    # 计算一下是否存在重复的embeddings
    embed_df = pd.DataFrame(embed)
    if embed_df.duplicated().any():
        print("%s-%d has duplicates" % (methodi, seedi))
        mask = embed_df.duplicated().values
        embed[mask, :] = embed[mask, :] + np.random.randn(mask.sum(), embed.shape[1]) * 1e-3
    key = "%s-%d" % (methodi, seedi)
    adata.obsm[key] = embed
    embed_keys.append(key)
print(embed_keys)

# %% [markdown] jp-MarkdownHeadingCollapsed=true
# ## scib_metrics

# %%
bm = Benchmarker(
    adata, batch_key="_batch", label_key=label_name, embedding_obsm_keys=embed_keys, n_jobs=8,
    # 其他方法的最优
    bio_conservation_metrics=BioConservation(nmi_ari_cluster_labels_kmeans=False, nmi_ari_cluster_labels_leiden=True)
)
bm.benchmark()

# %%
res_df = bm.get_results(min_max_scale=False, clean_names=True)
res_df = res_df.T
res_df[embed_keys] = res_df[embed_keys].astype(float)

res_df.to_csv(osp.join(res_root_compar, "benchmark_scib.csv"))

# %%
temp_df = res_df.drop(columns=["Metric Type"]).T
temp_df["method"] = temp_df.index.map(lambda x: x.split("-")[0])
plot_df = temp_df.groupby("method").mean().T
plot_df["Metric Type"] = res_df["Metric Type"]
plot_results_table(plot_df)
plt.show()

# %% [markdown]
# # Evaluate the mmAAVI

# %% [markdown] jp-MarkdownHeadingCollapsed=true
# ## metrics scores of multiple runs

# %%
# 查找所有的运行结果
runs = []
for cdim in range(18, 23):
    for di in os.listdir(root_mmAAVI):
        if di.startswith("cdimension_%d_" % cdim):
            runs.append((cdim, osp.join(root_mmAAVI, di)))
            break
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
# ## 选择cdimension

# %%
model_dir, nc = './res/2_mmAAVI/cdimension_21', 21

# %%
emb_keys = []
for i in os.listdir(model_dir):
    keyi = "mmAAVI-%s" % i
    emb_keys.append(keyi)

    model_dir_i = osp.join(model_dir, i)
    embed = torch.load(osp.join(model_dir_i, "latents.pt"))
    embed = embed.detach().cpu().numpy()
    adata.obsm[keyi] = embed

# %% [markdown] jp-MarkdownHeadingCollapsed=true
# ## scib_metrics

# %%
setup_seed(1234)
bm = Benchmarker(
    adata, batch_key="_batch", label_key=label_name, embedding_obsm_keys=emb_keys,
    bio_conservation_metrics=BioConservation(
        nmi_ari_cluster_labels_kmeans=False,
        nmi_ari_cluster_labels_leiden=True
    ), n_jobs=5
)
bm.benchmark()

# %%
res_df = bm.get_results(min_max_scale=False)
res_df = res_df.T
res_df[emb_keys] = res_df[emb_keys].astype(float)
res_df.to_csv(osp.join(res_root, "benchmark_scib_mmAAVI_nc%d.csv" % nc))

# %%
# 与之前计算好的comparison methods结果整合
res_df_others = pd.read_csv(osp.join(res_root_compar, "benchmark_scib.csv"), index_col=0)
res_df = pd.merge(res_df[emb_keys], res_df_others, how="outer", left_index=True, right_index=True)
metric_type = res_df["Metric Type"]
temp_df = res_df.drop(columns=["Metric Type"]).T
temp_df["method"] = temp_df.index.map(lambda x: x.split("-")[0])
plot_df = temp_df.groupby("method").mean().T
plot_df["Metric Type"] = res_df["Metric Type"]
plot_df.rename(columns=compar_methods_name, inplace=True)
plot_results_table(plot_df, save_name=osp.join(res_root, "scib_results_nc%d.svg" % nc))
plt.show()

# %%
plot_df

# %% [markdown]
# # UMAP Visualization

# %% [markdown]
# ## ALL

# %% [markdown]
# 把所有的方法都放在一起进行umap可视化

# %%
adata.obsm.keys()

# %%
umap_keys = ["%s-0" % mn for mn in compar_methods] + ["mmAAVI-2"]
methods_mapping = {}
for k in umap_keys:
    kk = k.split("-")[0]
    methods_mapping[k] = compar_methods_name.get(kk, kk)
methods_mapping

# %%
for k in tqdm(umap_keys):
    sc.pp.neighbors(adata, use_rep=k, key_added=k, n_neighbors=30)
    # sc.tl.umap(adata, neighbors_key=k, min_dist=0.2)

# %%
n_methods = len(umap_keys)
batch_uni = np.unique(adata.obs["_batch"].values)
cell_type_uni = np.unique(adata.obs[label_name].values)
# fine_label_uni = np.unique(adata.obs[fine_label].values)
markersize = 15000 / adata.n_obs
markerscale = 10. / markersize

batch_colors = sns.color_palette()
label_colors = sns.color_palette(cc.glasbey, n_colors=22)
# colors_fine = sns.color_palette(cc.glasbey, n_colors=12)


with warnings.catch_warnings(record=True):
    fig, axs = plt.subplots(ncols=2, nrows=n_methods, figsize=(2*4, n_methods*3.5))
    for i, key in tqdm(enumerate(umap_keys), total=n_methods):
        sc.tl.umap(adata, neighbors_key=key, min_dist=0.2)
        xy = pd.DataFrame(adata.obsm["X_umap"], columns=["UMAP1", "UMAP2"])
        xy["batch"] = adata.obs["_batch"].values
        xy["cell_type"] = adata.obs[label_name].values
        # xy["fine_label"] = adata.obs[fine_label].values

        ax = axs[i, 0]
        for j, batch_i in enumerate(batch_uni):
            xyi = xy[xy["batch"] == batch_i]
            ax.plot(xyi["UMAP1"], xyi["UMAP2"], ".", markersize=markersize, color=batch_colors[j], label=batch_i)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel("")
        ax.set_ylabel(methods_mapping[key])

        if i == (n_methods - 1):
            handles, labels = ax.get_legend_handles_labels()
            ax.legend(handles, labels, loc="upper center", markerscale=markerscale, frameon=False, fancybox=False, ncols=4,
                      bbox_to_anchor=(0.4, -0.2, 0.2, 0.2), columnspacing=0.2, handletextpad=0.1)

        ax = axs[i, 1]
        for j, ct_i in enumerate(cell_type_uni):
            xyi = xy[xy["cell_type"] == ct_i]
            ax.plot(xyi["UMAP1"], xyi["UMAP2"], ".", markersize=markersize, color=label_colors[j], label=ct_i)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel("")
        ax.set_ylabel("")
        if i == (n_methods - 1):
            handles, labels = ax.get_legend_handles_labels()
            ax.legend(handles, labels, loc="upper center", markerscale=markerscale, frameon=False, fancybox=False, ncols=4,
                      bbox_to_anchor=(0.4, -0.2, 0.1, 0.2), columnspacing=0.2, handletextpad=0.1)

        # ax = axs[i, 2]
        # for j, ct_i in enumerate(fine_label_uni):
        #     xyi = xy[xy["fine_label"] == ct_i]
        #     ax.plot(xyi["UMAP1"], xyi["UMAP2"], ".", markersize=markersize, color=colors_fine[j], label=ct_i)
        # ax.set_xticks([])
        # ax.set_yticks([])
        # ax.set_xlabel("")
        # ax.set_ylabel("")
        # if i == (n_methods - 1):
        #     handles, labels = ax.get_legend_handles_labels()
        #     ax.legend(handles, labels, loc="upper center", markerscale=markerscale, frameon=False, fancybox=False, ncols=4,
        #               bbox_to_anchor=(0.4, -0.2, 0.1, 0.2), columnspacing=0.2, handletextpad=0.1)

    axs[0, 0].set_title("Batch")
    axs[0, 1].set_title("Cell Type")
    # axs[0, 2].set_title("Fine-grained Cell Type")

    # fig.tight_layout()
plt.show()

# %%
fig.savefig(osp.join(res_root, "mop5b-umap.pdf"))
fig.savefig(osp.join(res_root, "mop5b-umap.png"), dpi=300)
fig.savefig(osp.join(res_root, "mop5b-umap.tiff"), dpi=300)

# %% [markdown]
# # 重复性（MOP5B）

# %%
df = pd.read_csv(osp.join(res_root, "benchmark_scib_mmAAVI_nc%d.csv" % nc), index_col=0)
plot_results_table(df, save_name=osp.join(res_root, "scib_results_mmAAVI_nc%d.svg" % nc))

# %% [markdown]
# 与其他方法的比较

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
