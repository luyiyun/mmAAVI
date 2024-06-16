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
# import warnings
import os
import os.path as osp

import numpy as np
import pandas as pd
import anndata as ad
import scanpy as sc
# import seaborn as sns
# import matplotlib.pyplot as plt
from scipy import sparse as sp
from tqdm import tqdm

# %matplotlib inline

# %%
sys.path.append(osp.abspath("../../src"))
from mmAAVI import dataset as D

# %%
root = "../../data/TripleOmics/"
res_dir = "./res/1_pp/"

# %%
os.makedirs(res_dir, exist_ok=True)

# %% [markdown]
# # 创建Mosaic数据集

# %% [markdown]
# ## read anndata

# %%
adata_atac = ad.read_h5ad(osp.join(root, "10x-ATAC-Brain5k.h5ad"))
adata_rna = ad.read_h5ad(osp.join(root, "Saunders-2018.h5ad"))
adata_methy = ad.read_h5ad(osp.join(root, "Luo-2017.h5ad"))

# %% [markdown]
# ## graph

# %%
atac_bed = D.Bed(adata_atac.var)
rna_bed = D.Bed(adata_rna.var)

atac_rna = atac_bed.window_graph(rna_bed.expand(upstream=2e3, downstream=0), window_size=0)
print(atac_rna.shape, atac_rna.nnz)

# %%
import re

pattern = re.compile(r"(.*?)_mC[HG]")
methy_var_genes = [pattern.search(namei).group(1) for namei in adata_methy.var_names]
print(methy_var_genes[:10])

row, col = [], []
for i, methyi in tqdm(enumerate(methy_var_genes), total=len(methy_var_genes)):
    for j, rnaj in enumerate(adata_rna.var_names):
        if methyi == rnaj:
            row.append(i)
            col.append(j)
methy_rna = sp.coo_array((np.ones(len(row)), (row, col)), shape=(len(methy_var_genes), adata_rna.shape[1]))
print(methy_rna.shape, methy_rna.nnz)

# %% [markdown]
# ## feature selection

# %% [markdown]
# * rna genes使用highly_variable
# * atac和methylation使用与rna genes存在连接的变量

# %%
mask_rna = adata_rna.var.highly_variable.values
mask_atac = (atac_rna.tocsr()[:, mask_rna].todense() != 0).any(axis=1)
mask_methy = (methy_rna.tocsr()[:, mask_rna].todense() != 0).any(axis=1)

print(mask_rna.sum(), mask_atac.sum(), mask_methy.sum())

# %%
adata_atac.var["highly_variable"] = mask_atac
adata_methy.var["highly_variable"] = mask_methy

# %% [markdown]
# ## create MosaicData

# %%
mdat = D.MosaicData([
    ("batch1", "rna", sp.csr_array(adata_rna.X[:, mask_rna])),
    ("batch2", "atac", sp.csr_array(adata_atac.X[:, mask_atac])),
    ("batch3", "met", adata_methy.layers["norm"][:, mask_methy]),  # 使用其中的norm layer，它才是服从log normal分布的
])
print(mdat)

# %% [markdown]
# ## generate representation of lsi_pca

# %% [markdown]
# 这里是glue的预处理流程

# %%
# RNA
# rna.layers["raw_count"] = rna.X.copy()
sc.pp.normalize_total(adata_rna)
sc.pp.log1p(adata_rna)
sc.pp.scale(adata_rna, max_value=10)
sc.tl.pca(adata_rna, n_comps=100, use_highly_variable=True, svd_solver="auto")

# %%
# methylation
adata_methy.layers["raw"] = adata_methy.X.copy()
adata_methy.X = adata_methy.layers["norm"].copy()
sc.pp.log1p(adata_methy)
sc.pp.scale(adata_methy, max_value=10)
sc.tl.pca(adata_methy, n_comps=100, use_highly_variable=True, svd_solver="auto")

# %%
# ATAC
arr_atac = D.lsi(sp.csr_array(adata_atac.X), n_components=100, n_iter=15)

# %%
mdat.reps["lsi_pca"] = [
    ("batch1", "rna", adata_rna.obsm["X_pca"]),
    ("batch2", "atac", arr_atac),
    ("batch3", "met", adata_methy.obsm["X_pca"])
]


# %% [markdown]
# ## generate representation of log1p_norm

# %%
# 对data_grid进行预处理，使用这些数据作为输入
def log1p_norm(i, j, dati):
    EPS = 1e-7

    if dati is None:
        return None

    if isinstance(dati, np.ndarray):
        dati = np.log1p(dati)
    elif sp.issparse(dati):
        dati = dati.log1p()
        dati = dati.todense()
    dati = (dati - dati.mean(axis=0, keepdims=True)) / \
        (dati.std(axis=0, keepdims=True) + EPS)
    return dati


# %%
mdat.reps["log1p_norm"] = mdat.X.apply(log1p_norm)

# %% [markdown]
# ## net:window

# %%
mdat.nets["window"] = [
    ("atac", "rna", atac_rna.tocsr()[adata_atac.var.highly_variable, :][:, adata_rna.var.highly_variable]),
    ("met", "rna", -methy_rna.tocsr()[adata_methy.var.highly_variable, :][:, adata_rna.var.highly_variable])  # methylation和rna的关系是负向的。
]

# %% [markdown]
# ## obs/var

# %% [markdown]
# * 只提取需要的变量。
# * 把各个组学的细胞类型分别使用不同的变量储存。
# * 创建一个coarse_cell_type，在三个组学间共享，用于后面评价使用。

# %%
adata_atac.obs["coarse_cell_type"] = adata_atac.obs.cell_type.replace({
    "L2/3 IT": "Layer2/3",
    "L4": "Layer5a", "L5 IT": "Layer5a", "L6 IT": "Layer5a",
    "L5 PT": "Layer5",
    "NP": "Layer5b",
    "L6 CT": "Layer6",
    "Vip": "CGE",
    "Pvalb": "MGE", "Sst": "MGE",
})

# %%
adata_methy.obs["coarse_cell_type"] = adata_methy.obs.cell_type.replace({
    "mL2/3": "Layer2/3",
    "mL4": "Layer5a", "mL5-1": "Layer5a", "mDL-1": "Layer5a", "mDL-2": "Layer5a",
    "mL5-2": "Layer5",
    "mL6-1": "Layer5b",
    "mL6-2": "Layer6",
    "mDL-3": "Claustrum",
    "mVip": "CGE", "mNdnf-1": "CGE", "mNdnf-2": "CGE",
    "mPv": "MGE", "mSst-1": "MGE", "mSst-2": "MGE"
})

# %%
obs_atac = adata_atac.obs[["cell_type"]].rename(columns={"cell_type": "cell_type_atac"})
obs_rna = adata_rna.obs[["cell_type"]].rename(columns={"cell_type": "cell_type_rna"})
obs_methy = adata_methy.obs[["cell_type"]].rename(columns={"cell_type": "cell_type_met"})
obs_all = pd.concat([obs_rna, obs_atac, obs_methy], axis=0)
obs_all["coarse_cell_type"] = pd.concat([adata_rna.obs.cell_type, adata_atac.obs.coarse_cell_type, adata_methy.obs.coarse_cell_type]).astype("category")
# 不加这个Others类别
# for i in range(obs_all.shape[1]):
#     obs_all.iloc[:, i] = obs_all.iloc[:, i].cat.add_categories("Others").fillna("Others")
obs_all.head()

# %%
var_use_cols = ["chrom", "chromStart", "chromEnd"]
var_all = []
for adatai, maski in zip([adata_rna, adata_atac, adata_methy], [mask_rna, mask_atac, mask_methy]):
    var_all.append(adatai.var.loc[maski, var_use_cols])
var_all = pd.concat(var_all, axis=0)
var_all.head()

# %%
mdat.obs = obs_all
mdat.var = var_all

# %% [markdown]
# ## 保存

# %%
print(mdat)
