import time
import logging
import os
import os.path as osp
from types import SimpleNamespace
from collections import defaultdict
from typing import Optional

import torch
import numpy as np
import pandas as pd
import scipy.sparse as sp
import scmomat

import utils as U


logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s][%(asctime)s]:%(message)s",
)


def run(
    data_fn: str,
    res_fn: str,
    model_fn: Optional[str] = None,
    use_pseduo: bool = False,
    K: int = 30,
    lamb: float = 0.001,  # weight on regularization term, default value
    T: int = 4000,  # number of total iterations, default value
    interval: int = 1000,  # print the result after each ``interval'' iterations, default value
    batch_size: float = 0.1,  # batch size for each iteraction, default value
    lr: float = 1e-2,  # learning rate, default value
    seed: int = 0,
    device: str = "cuda:0",
):
    device = torch.device(device)

    # 2. 读取数据
    logging.info("loading and preprocessing data...")
    _, var = U.read_obs_var(data_fn)
    feats_name = {}
    for ok in var["_omics"].unique():
        feats_name[ok] = var.query("_omics == '%s'" % ok).index.values

    # counts_atacs, counts_rnas, counts_proteins = [], [], []
    grid = U.read_data_grid(data_fn, "X", return_matrix=False)

    # 构造pseudo matrix
    if use_pseduo:
        net = U.read_nets(data_fn, return_matrix=False)["window"]
        A = net["atac"]["rna"].todense()
        for bk in grid.keys():
            grid_i = grid[bk]
            atac, rna = grid_i["atac"], grid_i["rna"]
            if atac is not None and rna is None:
                new_rna = ((atac @ A) != 0).astype(int)
                grid_i["rna"] = new_rna

    counts = defaultdict(list)
    for ok in ["atac", "rna"]:
        for dats in grid.values():
            dati = dats[ok]
            if sp.issparse(dati):
                dati = dati.todense()
            if dati is not None:
                if ok is not None:
                    if ok == "atac":
                        dati = scmomat.preprocess(dati, modality="ATAC")
                    else:
                        dati = scmomat.preprocess(
                            dati, modality="RNA", log=(ok == "protein")
                        )
            counts[ok].append(dati)

    counts["feats_name"] = feats_name
    counts["nbatches"] = len(grid)

    # 3. 运行模型，得到结果
    logging.info("running model...")
    logging.info("  training a new model.")
    start_time = time.time()
    model = scmomat.scmomat_model(
        counts=counts,
        K=K,
        batch_size=batch_size,
        interval=interval,
        lr=lr,
        lamb=lamb,
        seed=seed,
        device=device,
    )
    model.train_func(T=T)
    end_time = time.time()
    logging.info("  running time: " + str(end_time - start_time))
    if model_fn is not None:
        torch.save(model, model_fn)

    logging.info("  extract representation.")
    zs = model.extract_cell_factors()
    z = np.concatenate(zs)
    pd.DataFrame(z).to_csv(res_fn)


# 1. 设置路径
data_dir = "../res/1_pp/"
res_dir = "../res/3_comparison/scmomat/"
os.makedirs(res_dir, exist_ok=True)

# 2. run
for seedi in range(6):
    data_fn = osp.join(data_dir, "mop5b_full.mmod")
    res_fn = osp.join(res_dir, "mop5b_full_all_%d.csv" % seedi)
    print(res_fn)
    run(
        data_fn=data_fn,
        res_fn=res_fn,
        use_pseduo=True,
        K=30,
        seed=seedi,
        device="cuda:0",
    )
