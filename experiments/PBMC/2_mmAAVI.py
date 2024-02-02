import os
import os.path as osp

import logging

# import torch
# import numpy as np
import mudata as md
import scanpy as sc
from mmAAVI import MMAAVI
from mmAAVI.preprocess import merge_obs_from_all_modalities
# from scib_metrics.benchmark import Benchmarker


logging.basicConfig(level=logging.INFO)

data_dir, res_dir = "./data/", "./res/"
os.makedirs(res_dir, exist_ok=True)

mdata = md.read(osp.join(data_dir, "pbmc.h5mu"))
merge_obs_from_all_modalities(mdata, key="coarse_cluster")
merge_obs_from_all_modalities(mdata, key="batch")
print(mdata)
# batch1_indices = np.nonzero(mdata.obs["batch"] == 1)[0]
# label_indices = np.random.choice(batch1_indices, 100, replace=False)
# ss_label = np.full(mdata.n_obs, np.NaN, dtype=object)
# ss_label[label_indices] = mdata.obs["coarse_cluster"].iloc[label_indices]
# mdata.obs["semisup_label"] = ss_label

model = MMAAVI(
    input_key="log1p_norm",
    # sslabel_key="semisup_label",
    net_key="net",
    balance_sample="max",
    num_workers=4,
    hiddens_enc_c=(100, 50),
)
model.fit(mdata)
mdata.obs["mmAAVI_c_label"] = mdata.obsm["mmAAVI_c"].argmax(axis=1)

sc.pp.neighbors(mdata, use_rep="mmAAVI_z")
sc.tl.leiden(mdata, resolution=0.1, key_added="leiden")

model.differential(mdata, "leiden")

sc.tl.umap(mdata, min_dist=0.2)
# convert categorical
mdata.obs["batch"] = mdata.obs["batch"].astype("category")
mdata.obs["mmAAVI_c_label"] = mdata.obs["mmAAVI_c_label"].astype("category")
# plot and save umap
fig_umap = sc.pl.umap(
    mdata,
    color=["batch", "coarse_cluster", "mmAAVI_c_label", "leiden"],
    ncols=2,
    return_fig=True,
)
fig_umap.savefig(osp.join(res_dir, "umap.png"))
mdata.write(osp.join(res_dir, "pbmc_res.h5mu"))

# ------------ 测试 ------------
# ipdb.set_trace()
# data_dir = "./res/1_pp/"
# res_dir = "./res/2_mmAAVI/"
# data_fn = osp.join(data_dir, "pbmc.mmod")

# # running for different dimension of c
# for nc in [8]:  # range(3, 11):
#     print("nc = %d" % nc)
#     run(
#         dat=data_fn, nclusters=nc,
#         save_dir=osp.join(res_dir, "cdimension%d_" % nc),
#     )

# def run(
#     dat: Union[str, MosaicData],
#     dat_embed: Optional[Union[str, MosaicData]] = None,
#     save_dir: str = "./res/2_mmAAVI/pbmc_trial",
#     seeds: Sequence = (0, 1, 2, 3, 4, 5),
#     timing: bool = False,
#     nclusters: int = 8,
# ) -> None:
#     # loading preprocessed data
#     if isinstance(dat, str):
#         dat = MosaicData.load(dat)
#     elif not isinstance(dat, MosaicData):
#         raise ValueError("dat must be MosaicData "
#                          "or the file name of MosaicData")

#     if dat_embed is None:
#         dat_embed = dat
#     else:
#         if isinstance(dat_embed, str):
#             dat_embed = MosaicData.load(dat_embed)
#         elif not isinstance(dat_embed, MosaicData):
#             raise ValueError(
#                 "dat_embed must be MosaicData or the file name of MosaicData"
#             )

#     if timing:
#         res_timing = []

#     for seedi in seeds:
#         # set random seed
#         # deterministic保证重复性，但是性能慢两倍
#         setup_seed(seedi, deterministic=True)

#         # set path which contains model and results
#         save_dir_i = osp.join(save_dir, str(seedi))
#         os.makedirs(save_dir_i, exist_ok=True)
#         # print(save_dir)

#         # split dataset
#         tr_dat, va_dat = dat.split(0.1, strat_meta=None)

#         # prepare dataloaders
#         loaders = []
#         for k, dati in zip(
#             ["train", "valid", "test"], [tr_dat, va_dat, dat_embed]
#         ):
#             loaderi, in_dims, out_dims = MMAAVI.configure_data_to_loader(
#                 dati,
#                 drop_last=(k == "train"),  # 只有train drop last
#                 shuffle=False if k == "test" else True,
#                 resample="max" if k == "train" else None,
#                 num_workers=4,
#                 input_use="log1p_norm",
#                 output_use=None,
#                 net_use="window",
#             )
#             loaders.append(loaderi)
#         tr_loader, va_loader, loader = loaders

#         # construct model
#         model = MMAAVI.att_gmm_enc_model(
#             nbatches=tr_dat.nbatch, dim_inputs=in_dims, dim_outputs=out_dims,
#             nclusters=nclusters
#         )

#         # train model
#         if timing:
#             t1 = time.perf_counter()
#         hist_dfs, best_score = model.fit(
#             tr_loader,
#             va_loader,
#             device="cuda:0",
#             tensorboard_dir=osp.join(save_dir_i, "runs"),
#         )
#         if timing:
#             t2 = time.perf_counter()
#             res_timing.append(t2 - t1)

#         # encode and save encoded embeddings
#         embeds = model.encode(loader, device="cuda:0")

#         # save results of training
#         for k, dfi in hist_dfs.items():
#             dfi.set_index("epoch").to_csv(
#                 osp.join(save_dir_i, "hist_%s.csv" % k)
#             )
#         save_json(best_score, osp.join(save_dir_i, "best_score.csv"))

#         # save model
#         model.save(osp.join(save_dir_i, "model.pth"))

#         torch.save(embeds["z"], osp.join(save_dir_i, "latents.pt"))
#         if "c" in embeds:
#             torch.save(embeds["c"], osp.join(save_dir_i, "clusters.pt"))

#     if timing:
#         save_json(res_timing, osp.join(save_dir, "timing.json"))


# data_dir = "./res/1_pp/"
# res_dir = "./res/2_mmAAVI/"
# data_fn = osp.join(data_dir, "pbmc.mmod")

# # running for different dimension of c
# for nc in [8]:  # range(3, 11):
#     print("nc = %d" % nc)
#     run(
#         dat=data_fn, nclusters=nc,
#         save_dir=osp.join(res_dir, "cdimension%d_" % nc),
#     )

# # %%
# # running for subsampling experiments
# # data_fns = [
# #     osp.join(data_dir, fn)
# #     for fn in os.listdir(data_dir)
# #     if re.search(r"pbmc_[0-9]*?_[0-9].mmod", fn)
# # ]
# # for i, data_fni in enumerate(data_fns):
# #     # res_fn = osp.join(res_dir, "%s.csv" % osp.basename(data_fni)[:-5])
# #     prefix = "subample_%s_" % ("_".join(data_fni[:-5].split("_")[-2:]))
# #     print("%d/%d %s" % (i+1, len(data_fns), prefix))
# #     run(
# #         dat=data_fni,
# #         save_root=res_dir,
# #         trial_prefix=prefix,
# #         seeds=[0]
# #     )
