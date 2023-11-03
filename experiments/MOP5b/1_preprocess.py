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
from scipy import sparse as sp

# %matplotlib inline

# %%
sys.path.append(osp.abspath("../../src"))
from mmAAVI import dataset as D

# %%
root = "../../data/MOP5b/raw"
res_dir = "./res/1_pp/"

# %%
os.makedirs(res_dir, exist_ok=True)

# %% [markdown]
# # 读取数据创建数据集

# %%
counts_prefix = {"atac": "RxC", "rna": "GxC"}
# var_names = {"atac": "regions.txt", "rna": "genes.txt"}
batch_names = [str(i) for i in range(1, 6)]

# %% [markdown]
# ## var

# %%
# atac
fn = osp.join(root, "regions.txt")
names = np.loadtxt(fn, dtype="U")
dfi = pd.Series(names).str.split("_", expand=True)
assert not dfi.isna().any().any()
dfi.columns = ["chrom", "chromStart", "chromEnd"]
dfi["chrom"] = dfi["chrom"].str.slice(start=3)
dfi[["chromStart", "chromEnd"]] = \
    dfi[["chromStart", "chromEnd"]].astype(int)
dfi.index = names
var_atac = dfi
var_atac.head()

# %%
# rna
fn = osp.join(root, "genes.txt")
names = np.loadtxt(fn, dtype="U")
dfi = D.search_genomic_pos(
    names,
    remove_genes=(("ZNF781", 37668579.), ("LINC02256", 30427080)),
    cache_url="../mygene_cache"
)
# 将chrom的缺失值全部换成.，为了方便后面构造network
dfi.fillna({"chrom": "."}, inplace=True)
dfi["strand"] = dfi.strand.replace({1.0: "+", -1.0: "-"}).fillna("+")
dfi["chromStart"] = dfi.chromStart.fillna(0.)
dfi["chromEnd"] = dfi.chromEnd.fillna(0.)
var_rna = dfi
var_rna.head()

# %%
var = pd.concat([var_atac[[]], var_rna[[]]])

# %% [markdown]
# ## net:window

# %%
bed_rna = D.Bed(var_rna)
bed_atac = D.Bed(var_atac)

atac_rna = bed_atac.window_graph(
    bed_rna.expand(upstream=2e3, downstream=0),
    window_size=0, use_chrom=[str(i) for i in range(1, 23)] + ["X", "Y"]
)
print(atac_rna.shape, atac_rna.nnz)


# %% [markdown]
# ## X:counts

# %%
dats = []
for bi in batch_names:
    for oi, oprf in counts_prefix.items():
        fn = osp.join(root, "%s%s.npz" % (oprf, bi))
        if osp.exists(fn):
            datai = sp.load_npz(fn).T
            # 而对于这两种组学，使用稀疏矩阵更好（已测试）
            datai = sp.csr_array(datai)
        else:
            datai = None
        dats.append(("batch" + bi, oi, datai))
mdat = D.MosaicData(
    dats, var=var, nets={"window": [("atac", "rna", atac_rna)]}
)
print(mdat)

# %% [markdown]
# ## reps:log1p_norm

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
# ## reps:lsi_pca

# %%
def lsi_pca(bi, oi, x):
    if x is None:
        return None
    if oi == "atac":
        return D.lsi(x, n_components=100, n_iter=15)
    else:
        if sp.issparse(x):
            x = sp.csr_matrix(x)
        x = ad.AnnData(x).copy()
        sc.pp.normalize_total(x)
        sc.pp.log1p(x)
        sc.pp.scale(x)
        sc.tl.pca(x, n_comps=100, use_highly_variable=False, svd_solver="auto")
        return x.obsm["X_pca"]


# %%
mdat.reps["lsi_pca"] = mdat.X.apply(lsi_pca)

# %% [markdown]
# ## obs

# %%
obs = pd.concat([
    pd.read_csv(osp.join(root, "meta_c%s.csv" % bi), index_col=0)
    for bi in batch_names
])
obs.rename(columns={"cluster (remapped)": "cell_type"}, inplace=True)
obs = obs.loc[:, ["cell_type", "cluster", "region", "MajorCluster", "SubCluster"]]
print(obs.shape)
obs.head()

# %%
ind_dup = obs.index.duplicated(keep=False)
print("duplicated index has %d" % ind_dup.sum())
obs.reset_index(inplace=True)
obs.head()

# %%
mdat.obs = obs
# mdat.obs.head()

# %% [markdown]
# ## 保存

# %%
print(mdat)

# %%
mdat.save(osp.join(res_dir, "mop5b_full.mmod"))
