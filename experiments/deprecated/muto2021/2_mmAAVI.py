# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.14.6
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %%
import os
import os.path as osp
import sys
from copy import deepcopy
from datetime import datetime
from types import SimpleNamespace
from typing import Optional, Sequence, Union

import torch

# import numpy as np
# import pandas as pd

# %%
sys.path.append(osp.abspath("../../src/"))
from mmAAVI.dataset import MosaicData
from mmAAVI.model import MMAAVI
from mmAAVI.utils import save_json, setup_seed


# %%
# run函数
def run(
    cfg: SimpleNamespace,
    dat: Union[str, MosaicData],
    dat_embed: Optional[Union[str, MosaicData]] = None,
    save_root: str = "./res/2_mmAAVI/",
    trial_prefix: str = "",
    seeds: Sequence = (0, 1, 2, 3, 4, 5),
) -> None:
    trial_name = trial_prefix + datetime.today().strftime("%Y-%m-%d_%H-%M-%S")

    # loading preprocessed data
    if isinstance(dat, str):
        dat = MosaicData.load(dat)
    elif not isinstance(dat, MosaicData):
        raise ValueError(
            "dat must be MosaicData or the file name of MosaicData"
        )

    if dat_embed is None:
        dat_embed = dat
    else:
        if isinstance(dat_embed, str):
            dat_embed = MosaicData.load(dat_embed)
        elif not isinstance(dat_embed, MosaicData):
            raise ValueError(
                "dat_embed must be MosaicData or the file name of MosaicData"
            )

    for seedi in seeds:
        # configuration for this loop
        cfgi = deepcopy(cfg)
        cfgi.seed = seedi

        # set random seed
        # deterministic保证重复性，但是性能慢两倍
        setup_seed(cfgi.seed, deterministic=True)

        # set path which contains model and results
        save_dir = osp.join(save_root, trial_name, str(seedi))
        os.makedirs(save_dir, exist_ok=True)
        # print(save_dir)

        # save configuration
        save_json(cfgi.__dict__, os.path.join(save_dir, "config.json"))

        # split dataset
        tr_dat, va_dat = dat.split(cfgi.valid_size, strat_meta=None)

        if "batch_size" not in cfgi.loader:
            cfgi.loader["batch_size"] = (int(max(dat.nobs // 65, 32)),)  # 自动设置

        # prepare dataloaders
        loaders = []
        for k, dati in zip(
            ["train", "valid", "test"], [tr_dat, va_dat, dat_embed]
        ):
            loaderi, in_dims, out_dims = MMAAVI.configure_data_to_loader(
                dati,
                drop_last=(k == "train"),  # 只有train drop last
                shuffle=False if k == "test" else True,
                resample="max" if k == "train" else None,
                **cfgi.loader,
            )
            loaders.append(loaderi)
        tr_loader, va_loader, loader = loaders

        # construct model
        model = MMAAVI.att_gmm_enc_model(
            nbatches=tr_dat.nbatch,
            dim_inputs=in_dims,
            dim_outputs=out_dims,
            **cfgi.model_common,
            **cfgi.model_2,
        )

        # train model
        hist_dfs, best_score = model.fit(
            tr_loader,
            va_loader,
            tensorboard_dir=osp.join(save_dir, "runs"),
            **cfgi.train,
            **cfgi.weights,
        )

        # encode and save encoded embeddings
        embeds = model.encode(loader, device=cfgi.train["device"])

        # save results of training
        for k, dfi in hist_dfs.items():
            dfi.set_index("epoch").to_csv(
                osp.join(save_dir, "hist_%s.csv" % k)
            )
        save_json(best_score, osp.join(save_dir, "best_score.csv"))

        # save model
        model.save(osp.join(save_dir, "model.pth"))

        torch.save(embeds["z"], osp.join(save_dir, "latents.pt"))
        if "c" in embeds:
            torch.save(embeds["c"], osp.join(save_dir, "clusters.pt"))


# %%
# 一些配置
cfg = SimpleNamespace(
    seed=1234,
    valid_size=0.1,
    resample="max",
    loader=dict(
        batch_size=256,
        num_workers=4,
        input_use="lsi_pca",
        output_use=None,
        impute_miss=False,
        net_use="window_graph",
        # 2. gnn
        net_batch_size=10000,
        drop_self_loop=False,
        num_negative_samples=10,
    ),
    # common model config
    model_common={
        "dim_z": 30,
        "dim_u": 30,
        "nclusters": 8,
        "act": "lrelu",
        "reduction": "sum",
        "disc_condi_train": None,
        "input_with_batch": False,
        "distributions": "nb",
        "distributions_style": "batch",
        # "hiddens_disc": None,
        "hiddens_enc_c": [50],
        "hiddens_enc_u": [50],
        "hiddens_prior": [],
    },
    model_2=dict(
        disc_bn=True,
        graph_encoder_init_zero=True,
        c_reparam=True,
    ),
    train=dict(
        max_epochs=300,
        device="cuda:0",
        learning_rate=0.002,
        optimizer="rmsprop",
        valid_show_umap=None,  # ["_batch", "cell_type"],
        verbose=2,
        checkpoint_best=True,
        early_stop=True,
        lr_schedual="reduce_on_plateau",
        sch_kwargs={"factor": 0.1, "patience": 5},
        sch_max_update=2,
    ),
    weights=dict(
        alpha=20,
        weight_dot=0.99,
        weight_mlp=0.01,
        kl_c=1,
        kl_z=1,
        kl_u=1,
        label_smooth=0.1,
    ),
)

# %%
# setting the direction of saving
data_dir = "./res/1_pp/"
res_dir = "./res/2_mmAAVI/"


# %%
# running for different dimension of c
data_fn = osp.join(data_dir, "muto2021.mmod")
for nc in [10, 11, 12] + [14, 15, 16, 17]:
    print("nc = %d" % nc)
    cfg_cp = deepcopy(cfg)
    cfg_cp.loader["batch_size"] = 256
    cfg_cp.model_common["nclusters"] = nc
    cfg_cp.weights.update(
        {
            "weight_dot": 0.99,
            "weight_mlp": 0.01,
            "alpha": 50,
        }
    )
    cfg_cp.train.update(
        {
            "early_stop_patient": 20,
        }
    )
    run(
        cfg=cfg_cp,
        dat=data_fn,
        save_root=res_dir,
        trial_prefix="repeated_trial_%d_" % nc,
        seeds=range(6),
    )
