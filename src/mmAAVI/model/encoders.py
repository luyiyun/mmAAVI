import collections as ct
from abc import abstractmethod
from typing import Any, Dict, Optional, OrderedDict, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import (Categorical, Normal, RelaxedOneHotCategorical,
                                 kl_divergence)

from ..constant import EPS
from ..typehint import FORWARD, LOSS, T
from .block import (ConcatFusion, DistributionMLP, GatedAttentionFusion,
                    GraphConv, VanillaMLP)
from .utils import calc_kl_categorical, calc_kl_normal  # , focal_loss


class BaseEncoder(nn.Module):
    def __init__(self, reduction: str = "sum") -> None:
        super().__init__()
        if reduction == "sum":
            self.reductor = torch.sum
        elif reduction == "mean":
            self.reductor = torch.mean
        else:
            raise NotImplementedError

    @abstractmethod
    def forward(
        self,
        inpts: OrderedDict[str, T],
        masks: OrderedDict[str, T],
        covs: Sequence[T] = (),
        **kwargs: Any,
    ) -> FORWARD:
        """enocde"""

    @abstractmethod
    def step(
        self,
        inpts: OrderedDict[str, T],
        masks: OrderedDict[str, T],
        covs: Sequence[T] = (),
        **kwargs: Any,
    ) -> Tuple[FORWARD, LOSS]:
        """encode and calculate losses"""


""" Graph Encoders """


class Node2VecEncoder(nn.Module):
    EPS = 1e-15

    def __init__(self, num_nodes: int, embed_size: int) -> None:
        super().__init__()
        self.embedding_dim = embed_size
        self.embedding = nn.Embedding(num_nodes, embed_size)
        self.embedding.reset_parameters()

    def forward(self, node_index: Optional[T] = None) -> T:
        emb = self.embedding.weight
        return emb if node_index is None else emb.index_select(0, node_index)

    def step(self, pos_rw: T, neg_rw: T, pos_sign: Optional[T] = None) -> T:
        # Positive loss.
        start, rest = pos_rw[:, 0], pos_rw[:, 1:].contiguous()

        h_start = self.embedding(start).unsqueeze(1)  # n x 1 x embed_dim
        h_rest = self.embedding(rest)  # n x (l-1) x embed_dim
        out = (h_start * h_rest).sum(dim=-1)
        if pos_sign is not None:
            out = out * pos_sign
        pos_loss = -torch.log(torch.sigmoid(out.view(-1)) + self.EPS).mean()

        # Negative loss.
        start, rest = neg_rw[:, 0], neg_rw[:, 1:].contiguous()
        h_start = self.embedding(start).unsqueeze(1)  # n x 1 x embed_dim
        h_rest = self.embedding(rest)  # n x (l-1) x embed_dim
        out = (h_start * h_rest).sum(dim=-1).view(-1)
        neg_loss = -torch.log(1 - torch.sigmoid(out) + self.EPS).mean()

        return pos_loss + neg_loss


class GraphEncoder(BaseEncoder):
    r"""
    Graph encoder

    Parameters
    ----------
    vnum
        Number of vertices
    out_features
        Output dimensionality
    """

    def __init__(
        self,
        vnum: int,
        out_features: int,
        reduction: str = "sum",
        zero_init: bool = True,
    ) -> None:
        super().__init__(reduction)
        self.vrepr = nn.Parameter(torch.zeros(vnum, out_features))
        self.conv = GraphConv()
        self.loc = nn.Linear(out_features, out_features)
        self.std_lin = nn.Linear(out_features, out_features)
        if not zero_init:
            nn.init.trunc_normal_(self.vrepr, std=0.1)

    def forward(
        self, sidx: T, tidx: T, enorm: T, esgn: T
    ) -> Dict[str, Normal]:
        ptr = self.conv(self.vrepr, sidx, tidx, enorm, esgn)
        loc = self.loc(ptr)
        std = F.softplus(self.std_lin(ptr)) + EPS
        return {"z": Normal(loc, std)}

    def step(
        self, sidx: T, tidx: T, enorm: T, esgn: T
    ) -> Tuple[Dict[str, Normal], LOSS]:
        z = self.forward(sidx, tidx, enorm, esgn)
        loss_kl = calc_kl_normal(z["z"])
        loss_kl = self.reductor(loss_kl, dim=1).mean()  # TODO: /z.shape[0]
        return z, {"kl_graph": loss_kl}


""" Multi-Omics Encoder """


class PreCatEncoder(BaseEncoder):
    def __init__(
        self,
        incs: OrderedDict[str, int],
        outc: int,
        hiddens: Sequence[int],
        continue_cov_dims: list[int] = [],
        discrete_cov_dims: list[int] = [],
        act: str = "relu",
        bn: bool = True,
        dp: float = 0,
        reduction: str = "sum",
    ):
        super().__init__(reduction)
        self.model = ConcatFusion(
            incs,
            outc,
            hiddens,
            continue_cov_dims,
            discrete_cov_dims,
            act,
            bn,
            dp,
            last_bn=False,
            last_act=False,
            last_dp=False,
            distribution="normal",
            distribution_style="sample",
        )

    def forward(
        self,
        inpts: OrderedDict[str, T],
        masks: Optional[OrderedDict[str, T]] = None,
        continue_covs: list[T] = [],
        discrete_covs: list[T] = [],
    ) -> Dict[str, Normal]:
        return {"z": self.model(inpts, masks, continue_covs, discrete_covs)[1]}

    def step(
        self,
        inpts: OrderedDict[str, T],
        masks: Optional[OrderedDict[str, T]] = None,
        continue_covs: list[T] = [],
        discrete_covs: list[T] = [],
    ) -> Tuple[Dict[str, Normal], LOSS]:
        z = self.model(inpts, masks, continue_covs, discrete_covs)
        loss_kl = calc_kl_normal(z["z"])
        loss_kl = self.reductor(loss_kl, dim=1).mean()
        return z, {"kl": loss_kl}


class AttEncoder(BaseEncoder):
    def __init__(
        self,
        incs: OrderedDict[str, int],
        outc: int,
        continue_cov_dims: list[int] = [],
        discrete_cov_dims: list[int] = [],
        hiddens_unshare: Sequence[int] = (256,),
        nlatent_middle: int = 128,
        hiddens_shared: Sequence[int] = (256,),
        act: str = "relu",
        bn: bool = True,
        dp: float = 0,
        reduction: str = "sum",
        omic_embed: bool = False,
        omic_embed_train: bool = False,
    ):
        super().__init__(reduction)

        self.unshared_mlps = nn.ModuleDict()
        for name, ninpt in incs.items():
            self.unshared_mlps[name] = VanillaMLP(
                inc=ninpt,
                outc=nlatent_middle,
                hiddens=hiddens_unshare,
                continue_cov_dims=continue_cov_dims,
                discrete_cov_dims=discrete_cov_dims,
                act=act,
                bn=bn,
                dp=dp,
                last_act=True,
                last_bn=bn,
                last_dp=dp > 0,
            )

        fusion_inpt = ct.OrderedDict((k, nlatent_middle) for k in incs)
        self.fusion = GatedAttentionFusion(
            incs=fusion_inpt,
            omic_embed=omic_embed,
            omic_embed_train=omic_embed_train,
        )

        self.mapper = DistributionMLP(
            inc=nlatent_middle,
            outc=outc,
            hiddens=hiddens_shared,
            continue_cov_dims=[],
            discrete_cov_dims=[],
            act=act,
            bn=bn,
            dp=dp,
            last_act=False,
            last_bn=False,
            last_dp=False,
            distribution="normal",
            distribution_style="sample",
        )

    def forward(
        self,
        inpts: OrderedDict[str, T],
        masks: Optional[OrderedDict[str, T]] = None,
        continue_covs: list[T] = [],
        discrete_covs: list[T] = [],
    ) -> Dict[str, Normal]:
        embeds_dedicated = ct.OrderedDict()
        for k, mlpi in self.unshared_mlps.items():
            embeds_dedicated[k] = mlpi(inpts[k], continue_covs, discrete_covs)
        h, _ = self.fusion(embeds_dedicated, masks)
        _, z = self.mapper(h)
        return {"z": z}

    def step(
        self,
        inpts: OrderedDict[str, T],
        masks: Optional[OrderedDict[str, T]] = None,
        continue_covs: list[T] = [],
        discrete_covs: list[T] = [],
    ) -> Tuple[Dict[str, Normal], LOSS]:
        z = self.forward(inpts, masks, continue_covs, discrete_covs)
        loss_kl = calc_kl_normal(z["z"])
        loss_kl = self.reductor(loss_kl, dim=1).mean()
        return z, {"kl": loss_kl}

    def attention(
        self,
        inpts: OrderedDict[str, T],
        masks: Optional[OrderedDict[str, T]] = None,
        continue_covs: list[T] = [],
        discrete_covs: list[T] = [],
    ) -> T:
        embeds_dedicated = ct.OrderedDict()
        for k, mlpi in self.unshared_mlps.items():
            embeds_dedicated[k] = mlpi(inpts[k], continue_covs, discrete_covs)
        _, att = self.fusion(embeds_dedicated, masks)
        return att


class AttGMMEncoder(AttEncoder):
    def __init__(
        self,
        incs: OrderedDict[str, int],
        zdim: int,
        cdim: int,
        udim: int,
        continue_cov_dims: list[int] = [],
        discrete_cov_dims: list[int] = [],
        hiddens_unshare: Sequence[int] = (256,),
        nlatent_middle: int = 128,
        hiddens_z: Sequence[int] = (),
        hiddens_c: Sequence[int] = (),
        hiddens_u: Sequence[int] = (),
        hiddens_prior: Sequence[int] = (),
        act: str = "relu",
        bn: bool = True,
        dp: float = 0,
        reduction: str = "sum",
        omic_embed: bool = False,
        omic_embed_train: bool = False,
        c_reparam: bool = True,
        semi_supervised: bool = False,
        c_prior: Optional[Union[list[float], tuple[float], np.ndarray]] = None,
        # reuse_qcz_as_pcuz: bool = False,
    ):
        super().__init__(
            incs=incs,
            outc=zdim,
            continue_cov_dims=continue_cov_dims,
            discrete_cov_dims=discrete_cov_dims,
            hiddens_unshare=hiddens_unshare,
            nlatent_middle=nlatent_middle,
            hiddens_shared=hiddens_z,
            act=act,
            bn=bn,
            dp=dp,
            reduction=reduction,
            omic_embed=omic_embed,
            omic_embed_train=omic_embed_train,
        )

        self.cdim = cdim
        self.udim = udim
        self.c_reparam = c_reparam
        self.ssl = semi_supervised
        # self.reuse_qcz_as_pcuz = reuse_qcz_as_pcuz

        self.q_c_z = DistributionMLP(
            inc=zdim,
            outc=cdim,
            hiddens=hiddens_c,
            act=act,
            bn=bn,
            dp=dp,
            last_act=False,
            last_bn=False,
            last_dp=False,
            distribution="categorical_gumbel"
            if self.c_reparam
            else "categorical",
        )
        self.q_u_cz = DistributionMLP(
            inc=cdim + zdim,
            outc=udim,
            hiddens=hiddens_u,
            act=act,
            bn=bn,
            dp=dp,
            last_act=False,
            last_bn=False,
            last_dp=False,
            distribution="normal",
            distribution_style="sample",
        )
        self.p_z_cu = DistributionMLP(
            inc=cdim + udim,
            outc=zdim,
            hiddens=hiddens_prior,
            act=act,
            bn=bn,
            dp=dp,
            last_act=False,
            last_bn=False,
            last_dp=False,
            distribution="normal",
            distribution_style="sample",
        )

        self.register_buffer("diag", torch.eye(self.cdim, dtype=torch.float32))
        if c_prior is None:
            c_prior = torch.ones(cdim, dtype=torch.float32) / self.cdim
        else:
            c_prior = torch.tensor(c_prior, dtype=torch.float32)
        self.register_buffer("c_prior", c_prior)

    def forward(
        self,
        inpts: OrderedDict[str, T],
        masks: Optional[OrderedDict[str, T]] = None,
        continue_covs: list[T] = [],
        discrete_covs: list[T] = [],
        temperature: float = 1.0,
    ) -> Dict[str, Union[Normal, Categorical, RelaxedOneHotCategorical]]:
        z = super().forward(inpts, masks, continue_covs, discrete_covs)["z"]
        _, c = self.q_c_z(z.mean, temperature=temperature)
        return {"z": z, "c": c}

    def step(
        self,
        inpts: OrderedDict[str, T],
        masks: Optional[OrderedDict[str, T]] = None,
        continue_covs: list[T] = [],
        discrete_covs: list[T] = [],
        sslabel: Optional[T] = None,
        temperature: float = 1.0,
    ) -> Tuple[
        Dict[str, Union[T, Normal, Categorical, RelaxedOneHotCategorical]],
        LOSS,
    ]:
        fres = self.forward(
            inpts, masks, continue_covs, discrete_covs, temperature=temperature
        )
        z, c = fres["z"], fres["c"]
        zsample = z.rsample()
        logq_z_x = z.log_prob(zsample)

        if self.ssl:
            assert (
                sslabel is not None
            ), "must give sslabel when semi supervised learning"
            ssmask = sslabel != -1
            ssmask_ = ~ssmask
            zsample_ss = zsample[ssmask]
            zsample_us = zsample[ssmask_]
            logq_z_x_ss = logq_z_x[ssmask]
            logq_z_x_us = logq_z_x[ssmask_]
            ratio_ss = ssmask.float().mean()
            ratio_us = 1.0 - ratio_ss
            # flag_ssl = ssmask.any().item()
            flag_ssl = ratio_ss > 0.0
            flag_usl = ratio_us > 0.0
        else:
            zsample_us = zsample
            logq_z_x_us = logq_z_x
            ssmask_ = None
            flag_ssl = False
            flag_usl = True
            ratio_us = 1.0

        losses = {}

        if flag_ssl:
            # 计算c已知时的elbo，elbo=-logp(x)+kl(z)+kl(u)
            # -logp(x)将在decoder中进行计算，此处将计算剩下的2项
            sslabel_ss = sslabel[ssmask]
            sslabel_ss_oh = self.diag[sslabel_ss, :]
            _, u_ss = self.q_u_cz(
                torch.cat([sslabel_ss_oh, zsample_ss], dim=1)
            )
            usample_ss = u_ss.rsample()
            # 1. kl(z)
            _, z_cu = self.p_z_cu(
                torch.cat([sslabel_ss_oh, usample_ss], dim=-1)
            )  # N x (udim+cdim) -> N x zdim
            logp_z_uc = z_cu.log_prob(zsample_ss)  # N,zdim
            loss_ss_klz = self.reductor(logq_z_x_ss - logp_z_uc, dim=1).mean()
            # 3. kl(u)
            loss_ss_klu = calc_kl_normal(u_ss)
            loss_ss_klu = self.reductor(loss_ss_klu, dim=1).mean()

            # losses["rec_c"] = loss_ss_recc  # * ratio_ss
            losses["kl_z_ss"] = loss_ss_klz  # * ratio_ss
            losses["kl_u_ss"] = loss_ss_klu  # * ratio_ss

            # -------------------------------------------------------------------
            # supervised learning
            # 这里没有用到q_c_z，所以需要额外加一个监督训练项
            # -------------------------------------------------------------------
            loss_ss = F.nll_loss(c.logits[ssmask], sslabel_ss)
            # loss_ss = focal_loss(
            #     c.probs[ssmask], sslabel_ss, alpha=None, gamma=2.0
            # ).mean()
            losses["sup"] = loss_ss

        if flag_usl:
            if self.c_reparam:
                # 如果进行reparamatric
                csample = c.rsample()
                csample_us = csample[ssmask_] if self.ssl else csample
                _, u_us = self.q_u_cz(
                    torch.cat([csample_us, zsample_us], dim=1)
                )
                usample_us = u_us.rsample()

                _, z_prior = self.p_z_cu(
                    torch.cat([csample_us, usample_us], dim=1)
                )
                # NOTE: 因为对z进行了采样，kl_z这一项不是kl散度
                kl_z = self.reductor(
                    logq_z_x_us - z_prior.log_prob(zsample_us), dim=1
                ).mean()
                kl_u = self.reductor(calc_kl_normal(u_us), dim=1).mean()
            else:
                # 如果不进行reparamatric
                c_probs = c.probs
                c_probs_us = c_probs[ssmask_] if self.ssl else c_probs
                c_expand = self.diag.expand(zsample_us.size(0), -1, -1)

                _, u_us = self.q_u_cz(
                    torch.cat(
                        [
                            c_expand,
                            zsample_us.unsqueeze(1).expand(-1, self.cdim, -1),
                        ],
                        dim=-1,
                    )
                )  # N x cdim x (zdim+cdim) -> N x cdim x udim
                usample_us = u_us.rsample()  # N,cdim,udim

                _, z_prior = self.p_z_cu(
                    torch.cat([c_expand, usample_us], dim=-1)
                )  # N,cdim,(cdim+udim) -> N,cdim,zdim
                z_prior_p = z_prior.log_prob(
                    zsample_us[:, None, :]
                )  # N,cdim,zdim
                z_prior_p = (z_prior_p * c_probs_us[:, :, None]).sum(
                    dim=1
                )  # N,zdim
                kl_z = logq_z_x_us - z_prior_p
                # NOTE: 这个kl_z其实是个重构误差，即由z得到的c和u可以重新重构z

                kl_u = kl_divergence(
                    u_us,
                    Normal(
                        torch.zeros((1, 1, self.udim)).to(z_prior_p),
                        torch.ones((1, 1, self.udim)).to(z_prior_p),
                    ),
                )  # N,cdim,udim
                kl_u = (kl_u * c_probs_us[:, :, None]).sum(dim=1)

                kl_z = self.reductor(kl_z, dim=1).mean()
                kl_u = self.reductor(kl_u, dim=1).mean()
            kl_c = calc_kl_categorical(c, self.c_prior)
            kl_c = kl_c[ssmask_].mean()

            losses["kl_z"] = kl_z  # * ratio_us
            losses["kl_u"] = kl_u  # * ratio_us
            losses["kl_c"] = kl_c  # * ratio_us

        fres["zsample"] = zsample
        return fres, losses
