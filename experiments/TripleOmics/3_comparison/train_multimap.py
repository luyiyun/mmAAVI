import time
import logging
import os
import os.path as osp
# from types import SimpleNamespace
from typing import Optional, Sequence

import numpy as np
import pandas as pd
import scanpy as sc
import anndata
import MultiMAP

import utils as U


logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s][%(asctime)s]:%(message)s",
)


def run(
    data_fn: str,
    res_prefix: str,
    model_prefix: Optional[str] = None,
    K: int = 30,
    use_pseduo: bool = False,
    seeds: Sequence[int] = [0],
):
    # 2. 读取数据
    logging.info("loading and preprocessing data...")
    obs, var = U.read_obs_var(data_fn)
    feats_name = {}
    for ok in var["_omics"].unique():
        feats_name[ok] = var.query("_omics == '%s'" % ok).index.values

    # counts_atacs, counts_rnas, counts_proteins = [], [], []
    grid = U.read_data_grid(data_fn, "X", return_matrix=False)

    # 构造pseudo matrix
    if use_pseduo:
        net = U.read_nets(data_fn, return_matrix=False)["window"]
        A1 = net["atac"]["rna"].todense()
        A2 = net["met"]["rna"].todense()
        for bk in grid.keys():
            grid_i = grid[bk]
            atac, rna = grid_i["atac"], grid_i["rna"]
            met = grid_i["met"]
            if atac is not None and rna is None:
                new_rna = ((atac @ A1) != 0).astype(int)
                grid_i["rna"] = new_rna
            if met is not None and rna is None:
                new_rna = ((met @ A2) != 0).astype(int)
                grid_i["rna"] = new_rna

    # 3. multimap需要重新组织一下数据格式
    # 直接用int index，避免重复变量名带来的问题
    feats_name = {
        k: np.array(["%s_%d" % (k, i) for i in range(v.shape[0])])
        for k, v in feats_name.items()
    }

    adatas_multi_omics = []
    for k, dats in grid.items():
        obs_k = obs.query("_batch == '%s'" % k)
        adatas = []
        for k in ["atac", "rna", "met"]:
            dati = dats[k]
            if dati is not None:
                adatai = anndata.AnnData(
                    X=dati, obs=obs_k,
                    var=pd.DataFrame(index=feats_name[k])
                )
                adatas.append(adatai)
        adatas = anndata.concat(adatas, axis=1)
        adatas_multi_omics.append(adatas)

    for i, ci in enumerate(adatas_multi_omics):
        logging.info("  preprocessing batch %d ..." % (i + 1))
        sc.pp.log1p(ci)
        sc.pp.scale(ci)
        if i == 2:  # 不然，这些值是0，会导致pca无法进行
            ci.X[:, ci.var["std"].isna().values] = 0.
        sc.pp.pca(ci)

    # 4. 运行模型，得到结果
    for seedi in seeds:
        logging.info("running model for random seed = %d ..." % seedi)
        start_time = time.time()
        adata = MultiMAP.Integration(
            adatas=adatas_multi_omics,
            use_reps=["X_pca"] * len(adatas_multi_omics),
            n_components=K,
            seed=seedi
        )
        end_time = time.time()
        logging.info("  running time: " + str(end_time - start_time))
        if model_prefix is not None:
            adata.write_h5ad("%s_%d.h5" % (model_prefix, seedi))

        z = adata.obsm["X_multimap"]
        pd.DataFrame(z).to_csv("%s_%d.csv" % (res_prefix, seedi))


# 1. 设置路径
data_dir = "../res/1_pp/"
res_dir = "../res/3_comparison/multimap/"
os.makedirs(res_dir, exist_ok=True)

# 2. run
data_fn = osp.join(data_dir, "TripleOmics.mmod")
res_prefix = osp.join(res_dir, "TripleOmics_all")
run(
    data_fn=data_fn,
    res_prefix=res_prefix,
    use_pseduo=True,
    K=30,
    seeds=range(6),
)
