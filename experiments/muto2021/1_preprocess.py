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
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import sparse as sp

# %matplotlib inline

# %%
sys.path.append(osp.abspath("../../src"))
from mmAAVI import dataset as D

# %%
root = "../../data/muto-2021/"
res_dir = "./res/1_pp/"

# %%
os.makedirs(res_dir, exist_ok=True)

# %%
# 一些配置
select_by_graph = {"atac": True, "rna": False}
select_by_highly_var = {"atac": True, "rna": True}

# %% [markdown]
# # 读取数据

# %%
atac = ad.read_h5ad(osp.join(root, "Muto-2021-ATAC.h5ad"))
atac

# %%
rna = ad.read_h5ad(osp.join(root, "Muto-2021-RNA.h5ad"))
rna

# %% [markdown]
# # 创建数据集

# %% [markdown]
# ## 预处理

# %% [markdown]
# 仿效glue，得到pca和lsi的结果

# %% [markdown]
# ### RNA

# %%
rna.layers["counts"] = rna.X.copy()
sc.pp.normalize_total(rna)
sc.pp.log1p(rna)
sc.pp.scale(rna, max_value=10)
sc.tl.pca(rna, n_comps=100, svd_solver="auto")

# %% [markdown]
# ### ATAC

# %%
import sklearn
import scipy
from sklearn.preprocessing import normalize


def tfidf(X):
    r"""
    TF-IDF normalization (following the Seurat v3 approach)

    Parameters
    ----------
    X
        Input matrix

    Returns
    -------
    X_tfidf
        TF-IDF normalized matrix
    """
    idf = X.shape[0] / X.sum(axis=0)
    if scipy.sparse.issparse(X):
        tf = X.multiply(1 / X.sum(axis=1))
        return tf.multiply(idf)
    else:
        tf = X / X.sum(axis=1, keepdims=True)
        return tf * idf


def lsi(adata, n_components=20, use_highly_variable=None, **kwargs) -> None:
    r"""
    LSI analysis (following the Seurat v3 approach)

    Parameters
    ----------
    adata
        Input dataset
    n_components
        Number of dimensions to use
    use_highly_variable
        Whether to use highly variable features only, stored in
        ``adata.var['highly_variable']``. By default uses them if they
        have been determined beforehand.
    **kwargs
        Additional keyword arguments are passed to
        :func:`sklearn.utils.extmath.randomized_svd`
    """
    if "random_state" not in kwargs:
        kwargs["random_state"] = 0  # Keep deterministic as the default behavior
    if use_highly_variable is None:
        use_highly_variable = "highly_variable" in adata.var
    adata_use = adata[:, adata.var["highly_variable"]] if use_highly_variable else adata
    X = tfidf(adata_use.X)
    X_norm = normalize(X, norm="l1")
    X_norm = np.log1p(X_norm * 1e4)
    X_lsi = sklearn.utils.extmath.randomized_svd(X_norm, n_components, **kwargs)[0]
    X_lsi -= X_lsi.mean(axis=1, keepdims=True)
    X_lsi /= X_lsi.std(axis=1, ddof=1, keepdims=True)
    adata.obsm["X_lsi"] = X_lsi


# %%
atac.layers["counts"] = atac.X.copy()
lsi(atac, n_components=100, use_highly_variable=False, n_iter=15)

# %% [markdown]
# ## 构建MosaicData

# %% [markdown]
# ### variables selection

# %%
if select_by_highly_var["atac"]:
    # 对于ATAC，需要首先进行一些变量筛选
    sc.pp.highly_variable_genes(atac, flavor="cell_ranger")
else:
    atac.var["highly_variable"] = True
print(atac.var.highly_variable.sum())

# %%
if select_by_highly_var["rna"]:
    pass
else:
    rna.var["highly_variable"] = True
print(rna.var.highly_variable.sum())

# %% [markdown]
# ### graph

# %%
bed_rna = D.Bed(rna.var)
bed_atac = D.Bed(atac.var)

atac_rna = bed_atac.window_graph(bed_rna.expand(upstream=2e3, downstream=0), window_size=0)
print(atac_rna.shape, atac_rna.nnz)

# %%
# if any(select_by_graph.values()):
atac_rna = atac_rna.todense()
mask = atac_rna > 0.
if select_by_graph["atac"]:
    mask_atac = mask.any(axis=1)
    atac.var["highly_variable"] = np.logical_and(atac.var.highly_variable, mask_atac)
if select_by_graph["rna"]:
    mask_rna = mask.any(axis=0)
    atac_rna = atac_rna[:, mask_rna]
    rna.var["highly_variable"] = np.logical_and(rna.var.highly_variable, mask_rna)

atac_rna = sp.coo_array(atac_rna[atac.var.highly_variable.values, :][:, rna.var.highly_variable.values])

# %%
print(atac.var.highly_variable.sum(), rna.var.highly_variable.sum(), atac_rna.shape)

# %% [markdown]
# ### create MosaicData

# %%
grids, obs, rep_lsi_pca = [], [], []
batch_index = list(set(atac.obs.batch.cat.categories).union(rna.obs.batch.cat.categories))
for bi in batch_index:
    for k, datai in zip(["atac", "rna"], [atac, rna]):
        mask_i = datai.obs.batch == bi
        if mask_i.any():
            data_sub = datai[mask_i, datai.var.highly_variable]
            obs.append(data_sub.obs)
            grids.append((k+"_"+bi, k, sp.csr_array(data_sub.layers["counts"])))
            rep_lsi_pca.append((k+"_"+bi, k, np.asarray(data_sub.obsm["X_pca"] if k == "rna" else data_sub.obsm["X_lsi"])))

# %%
mdata = D.MosaicData(
    data=grids,
    obs=pd.concat(obs, axis=0, ignore_index=True),
    var=pd.concat([atac.var.loc[atac.var.highly_variable],
                   rna.var.loc[rna.var.highly_variable]], axis=0, ignore_index=True),
    nets={"window_graph": [("atac", "rna", atac_rna)]},
    reps={"lsi_pca": rep_lsi_pca},
)


# %% [markdown]
# ### generate representation by log1p-norm

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
mdata.reps["log1p_norm"] = mdata.X.apply(log1p_norm)


# %%
# 设置一个特殊的batch
mdata.obs["batch5"] = mdata.obs["_batch"].map(lambda x: x.split("_")[1])
mdata.obs["omic"] = mdata.obs["_batch"].map(lambda x: x.split("_")[0])
mdata.reps["log1p_norm"] = mdata.X.apply(log1p_norm)

# %% [markdown]
# ### saving

# %%
print(mdata)

# %%
mdata.save(osp.join(res_dir, "muto2021.mmod"))

# %%
