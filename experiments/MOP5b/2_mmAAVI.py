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
from typing import Optional, Sequence, Union

import torch

# %%
sys.path.append(osp.abspath("../../src/"))
from mmAAVI.dataset import MosaicData
from mmAAVI.model import MMAAVI
from mmAAVI.utils import setup_seed, save_json


# %%
# run函数
def run(
    dat: Union[str, MosaicData],
    dat_embed: Optional[Union[str, MosaicData]] = None,
    # save_root: str = "./res/2_mmAAVI/",
    # trial_prefix: str = "",
    save_dir_i: str = "./res/2_mmAAVI/pbmc_trial",
    seeds: Sequence = (0, 1, 2, 3, 4, 5),
    nclusters: int = 21,
) -> None:
    # loading preprocessed data
    if isinstance(dat, str):
        dat = MosaicData.load(dat)
    elif not isinstance(dat, MosaicData):
        raise ValueError("dat must be MosaicData "
                         "or the file name of MosaicData")

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
        # set random seed
        # deterministic保证重复性，但是性能慢两倍
        setup_seed(seedi, deterministic=True)

        # set path which contains model and results
        save_dir_i = osp.join(save_dir_i, str(seedi))
        os.makedirs(save_dir_i, exist_ok=True)
        # print(save_dir)

        # split dataset
        tr_dat, va_dat = dat.split(0.1, strat_meta=None)

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
                num_workers=4,
                input_use="log1p_norm",
                output_use=None,
                net_use="window",
            )
            loaders.append(loaderi)
        tr_loader, va_loader, loader = loaders

        # construct model
        model = MMAAVI.att_gmm_enc_model(
            nbatches=tr_dat.nbatch, dim_inputs=in_dims, dim_outputs=out_dims,
            nclusters=nclusters
        )

        # train model
        hist_dfs, best_score = model.fit(
            tr_loader,
            va_loader,
            device="cuda:0",
            tensorboard_dir=osp.join(save_dir_i, "runs"),
        )

        # encode and save encoded embeddings
        embeds = model.encode(loader, device="cuda:0")

        # save results of training
        for k, dfi in hist_dfs.items():
            dfi.set_index("epoch").to_csv(
                osp.join(save_dir_i, "hist_%s.csv" % k)
            )
        save_json(best_score, osp.join(save_dir_i, "best_score.csv"))

        # save model
        model.save(osp.join(save_dir_i, "model.pth"))

        torch.save(embeds["z"], osp.join(save_dir_i, "latents.pt"))
        if "c" in embeds:
            torch.save(embeds["c"], osp.join(save_dir_i, "clusters.pt"))


# %%
# setting the direction of saving
data_dir = "./res/1_pp/"
res_dir = "./res/2_mmAAVI/"

# %%
# running for different dimension of c
data_fn = osp.join(data_dir, "mop5b_full.mmod")
for nc in range(18, 23):
    print("nc = %d" % nc)
    run(
        dat=data_fn, nclusters=nc,
        save_dir=osp.join(res_dir, "cdimension%d_" % nc),
    )
