#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
from datetime import datetime
from typing import Optional

import torch

from ..src import dataset as D
from ..src.model import MMOATTGMVAE
from ..src.utils import setup_seed, save_json
# sys.path.append("/home/rong/Documents/mosaic-GAN/")
# from src.dataset import MosaicMultiOmicsDataset
# from src.model import MMOATTGMVAE
# from src.utils import setup_seed, save_json


os.chdir("./experiments/muto2021/")



# # 配置

# In[3]:
FLAG = False


from dataclasses import dataclass, asdict
from typing import Optional

@dataclass
class Config:
    seed: int = 1234
    # dataset
    fn: str = "./res/1_pp/muto2021.mmod"
    label_name: str = "cell_type"
    input_use: Optional[str] = "log1p_norm"
    output_use: Optional[str] = None
    batch_size: int = 512
    num_workers: int = 8
    drop_last: bool = False
    stratify: bool = True
    train_with_valid: bool = True
    valid_size: float = 0.1
    remove_block: Optional[int] = None
    # model
    ncluster: int = 12
    zdim: int = 30
    udim: int = 30
    hiddens_dedicated: tuple[int] = (256, 256)
    hiddens_encoder_z: tuple[int] = (256,)
    hiddens_encoder_c: tuple[int] = (256,)
    hiddens_encoder_u: tuple[int] = (256,)
    hiddens_disc: tuple[int] = (256, 256)
    hiddens_decoder: tuple[int] = (256, 256)
    dist: str | dict[str, str] = "NB4"  # NB2train不动
    bn: bool = True
    act: str = "relu"
    dp: float = 0.2
    disc_condi: Optional[str] = "hard"
    disc_use_mean: bool = False
    c_reparam: bool = False
    network_constraint: Optional[str] = "lproj"
    nlatent_dedicated: int = 200
    temperature: float = 1.0
    alpha: float = 30
    lam_rec: float = 1.
    lam_kl_c: float = 0.5
    lam_kl_u: float = 0.5
    lam_kl_z: float = 0.5
    lam_disc: float = 1.
    lam_net: float = 0.05
    # fit
    device: str = "cuda:0"
    max_epochs: int = 300
    optimizer: str = "rmsprop"
    learning_rate: float = 0.002
    weight_decay: float = 0.0
    grad_clip: Optional[float] = None
    valid_interval: int = 3
    valid_calc_scores: bool = True
    valid_show_umap: bool = True
    checkpoint_best: bool = False
    checkpoint_score: str = "valid/total"
    early_stop: bool = False
    early_stop_patient: int = 10

cfg = Config()


# In[4]:


# 查看kl系数的变化趋势
# x = np.linspace(0, 299, 1000)
# y = kl_c(x)
# sns.relplot(x=x, y=y, kind="line", aspect=2, height=2)
# plt.show()


# # 保存配置

# In[5]:


save_name = "./res/2_mmAAVI/%s" % datetime.today().strftime("%Y-%m-%d_%H-%M-%S")
os.makedirs(save_name, exist_ok=True)
print(save_name)


# In[6]:


# OmegaConf.save(cfg, os.path.join(save_name, "config.yaml"))
save_json(asdict(cfg), os.path.join(save_name, "config2.json"))


# # 设置随机种子

# In[7]:


setup_seed(cfg.seed)


# # 准备数据

# In[8]:


# 载入预处理好的数据
dat = MosaicMultiOmicsDataset.load(cfg.fn)
print(dat)


# In[9]:


# 是否需要去掉某个block
# if cfgd.remove_block is not None:
#     dat = dat.remove_block(*cfgd.remove_block)
# 数据集分割
tr_dat, va_dat = dat.split(
    cfg.valid_size,
    strat_meta=cfg.label_name if cfg.stratify else None,
    seed=cfg.seed
)
if cfg.train_with_valid:
    tr_dat = dat
# 得到dataloaders
tr_dat.prepare(input_use=cfg.input_use, output_use=cfg.output_use, label_name=cfg.label_name)
va_dat.prepare(input_use=cfg.input_use, output_use=cfg.output_use, label_name=cfg.label_name)
dat.prepare(input_use=cfg.input_use, output_use=cfg.output_use, label_name=cfg.label_name)


# # 构建模型

# In[10]:


model = MMOATTGMVAE(
    dat.nobs,
    nclusters=cfg.ncluster,
    dim_inputs=dat.var_dims(cfg.input_use),
    dim_outputs=dat.var_dims(cfg.output_use),
    dim_z=cfg.zdim,
    dim_u=cfg.udim,
    hiddens_enc_unshared=cfg.hiddens_dedicated,
    hiddens_enc_z=cfg.hiddens_encoder_z,
    hiddens_enc_c=cfg.hiddens_encoder_c,
    hiddens_enc_u=cfg.hiddens_encoder_u,
    hiddens_dec=cfg.hiddens_decoder,
    hiddens_disc=cfg.hiddens_disc,
    distributions=cfg.dist,
    bn=cfg.bn,
    act=cfg.act,
    dp=cfg.dp,
    disc_condi_train=cfg.disc_condi,
    disc_use_mean=cfg.disc_use_mean,
    c_reparam=cfg.c_reparam,
    network_constraint=cfg.network_constraint,
    dim_enc_dedicated=cfg.nlatent_dedicated,
    alpha=cfg.alpha,
    temperature=cfg.temperature,
    lam_rec=cfg.lam_rec,
    lam_kl_z=cfg.lam_kl_z,
    lam_kl_c=cfg.lam_kl_c,
    lam_kl_u=cfg.lam_kl_u,
    lam_disc=cfg.lam_disc,
    lam_net=cfg.lam_net,
)


# # 训练模型

# In[11]:


# 训练模型
hist_dfs, best_score = model.fit(
    tr_dat, va_dat,
    batch_size=cfg.batch_size,
    num_workers=cfg.num_workers,
    max_epochs=cfg.max_epochs,
    device=cfg.device,
    learning_rate=cfg.learning_rate,
    optimizer=cfg.optimizer,
    weight_decay=cfg.weight_decay,
    grad_clip=cfg.grad_clip,
    # lr_schedual=cfgt.lr_schedual,
    # sch_kwargs=None,
    valid_interval=cfg.valid_interval,
    valid_calc_scores=cfg.valid_calc_scores,
    valid_show_umap=cfg.valid_show_umap,
    checkpoint_best=cfg.checkpoint_best,
    checkpoint_score=cfg.checkpoint_score,
    tensorboard_dir=os.path.join(save_name, "runs"),
    verbose=2,
    early_stop=cfg.early_stop,
    early_stop_patient=cfg.early_stop_patient,
)


# In[ ]:


for k, dfi in hist_dfs.items():
    dfi.set_index("epoch").to_csv(os.path.join(save_name, "hist_%s.csv" % k))
save_json(best_score, os.path.join(save_name, "best_score.csv"))
torch.save(model.state_dict(), os.path.join(save_name, "model.pth"))


# # 预测结果

# In[ ]:


z, clu_prob = model.encode(dat, batch_size=cfg.batch_size, num_workers=cfg.num_workers, device=cfg.device)
torch.save(z, os.path.join(save_name, "predictions.pt"))
if clu_prob is not None:
    torch.save(clu_prob, os.path.join(save_name, "clusters.pt"))


# In[ ]:
