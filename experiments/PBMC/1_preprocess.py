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

import numpy as np
import pandas as pd
import anndata as ad
import scanpy as sc
from scipy import sparse as sp
from tqdm import tqdm

# %%
sys.path.append(osp.abspath("../../src"))
from mmAAVI import dataset as D

# %%
root = "../../data/PBMC/raw"
res_dir = "./res/1_pp/"

# %%
os.makedirs(res_dir, exist_ok=True)

# %% [markdown]
# # Create the mosaic dataset

# %%
counts_prefix = {"atac": "RxC", "rna": "GxC", "protein": "PxC"}
batch_names = [str(i) for i in range(1, 5)]

# %% [markdown]
# ## variables

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
# protein
fn = osp.join(root, "proteins_alias.txt")
names = np.loadtxt(fn, dtype="U")
names_ori = [line[0] for line in names]
names_alias = [",".join(line) for line in names]
var_protein = pd.DataFrame(dict(alias=names_alias), index=names_ori)
var_protein.head()

# %%
var = pd.concat([var_atac[[]], var_rna[[]], var_protein[[]]])
var.tail()

# %% [markdown]
# ## graph

# %%
# atac-rna

bed_rna = D.Bed(var_rna)
bed_atac = D.Bed(var_atac)

atac_rna = bed_atac.window_graph(
    bed_rna.expand(upstream=2e3, downstream=0),
    window_size=0, use_chrom=[str(i) for i in range(1, 23)]+["X","Y"]
)
print(atac_rna.shape, atac_rna.nnz)

# %%
# rna-protein
row, col = [], []
for i, p_alias in tqdm(enumerate(var_protein["alias"]), total=var_protein.shape[0]):
    for j, g_symbol in enumerate(var_rna.index):
        if (g_symbol in p_alias) or (g_symbol.lower() in p_alias.lower()):
            row.append(j)
            col.append(i)
row, col = np.array(row), np.array(col)
rna_protein = sp.coo_array((np.ones_like(row), (row, col)), shape=(var_rna.shape[0], var_protein.shape[0]))
print(rna_protein.shape)
print(rna_protein.nnz)

# %%
# only remain the features which are in the network
mask_atac = (atac_rna.todense() > 0.).any(axis=1)
mask_rna = (atac_rna.todense() > 0.).any(axis=0) | (rna_protein.todense() > 0.).any(axis=1)
mask_protein = (rna_protein.todense() > 0.).any(axis=0)
print(mask_atac.sum(), mask_rna.sum(), mask_protein.sum())

var = pd.concat([var_atac.loc[mask_atac, []], var_rna.loc[mask_rna, []], var_protein.loc[mask_protein, []]])
atac_rna = sp.coo_array(atac_rna.todense()[mask_atac, :][:, mask_rna])
rna_protein = sp.coo_array(rna_protein.todense()[mask_rna, :][:, mask_protein])

# %% [markdown]
# ## expressions

# %%
dats = []
for bi in batch_names:
    for oi, oprf in counts_prefix.items():
        fn = osp.join(root, "%s%s.npz" % (oprf, bi))
        if osp.exists(fn):
            datai = sp.load_npz(fn).T
            if oi == "protein":
                datai = np.array(datai.todense())
            else:
                datai = sp.csr_array(datai)
            feat_mask = {"atac": mask_atac, "rna": mask_rna, "protein": mask_protein}[oi]
            datai = datai[:, feat_mask]
        else:
            datai = None
        dats.append(("batch"+bi, oi, datai))
mdat = D.MosaicData(
    dats, var=var, nets={"window": [("atac", "rna", atac_rna), ("rna", "protein", rna_protein)]}
)

# %%
print(mdat)


# %% [markdown]
# ## representations

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
# ## observations

# %%
obs = pd.concat([
    pd.read_csv(osp.join(root, "meta_c%s.csv" % bi), index_col=0)
    for bi in batch_names
])
obs.rename(columns={"coarse_cluster": "cell_type", "cluster": "fine_cell_type"}, inplace=True)
print(obs.shape)
obs.head()

# %%
ind_dup = obs.index.duplicated(keep=False)
print("duplicated index has %d" % ind_dup.sum())
obs.reset_index(inplace=True)
obs.head()

# %%
mdat.obs = obs

# %% [markdown]
# ## saving

# %%
print(mdat)

# %%
mdat.save(osp.join(res_dir, "pbmc.mmod"))

# %% [markdown]
# # Subsample dataset

# %%
mdat = D.MosaicData.load(osp.join(res_dir, "pbmc.mmod"))
print(mdat)

# %%
subsample_sizes = np.arange(0.9, 0.0, -0.1)
seeds = list(range(6))

# %%
for subsize in subsample_sizes:
    for seedi in seeds:
        _, dat_used = mdat.split(subsize, seed=seedi)
        dat_used.save(osp.join(res_dir, "pbmc_%d_%d.mmod" % (dat_used.nobs, seedi)))
