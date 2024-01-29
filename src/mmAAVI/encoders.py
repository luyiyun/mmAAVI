from abc import abstractmethod
from typing import (
    Any,
    Dict,
    Optional,
    Sequence,
    Tuple,
    Union,
)

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import (
    Normal,
    kl_divergence,
    register_kl,
    Categorical,
    RelaxedOneHotCategorical,
)

from .dataset import GMINIBATCH
from .model.block import (
    VanillaMLP,
    GatedAttentionFusion,
    DistributionMLP,
    GraphConv,
)

T = torch.Tensor
FRES = Dict[str, Any]  # forward results
LOSS = Dict[str, T]


def calc_kl_normal(z: Normal) -> T:
    mu, sigma2 = z.mean, z.variance
    prior = Normal(torch.zeros_like(mu), torch.ones_like(sigma2))
    kl = kl_divergence(z, prior)
    return kl


def calc_kl_categorical(c: Categorical, prior_c: Optional[T] = None) -> T:
    if prior_c is None:
        prior = Categorical(logits=torch.ones_like(c.logits))
    else:
        post_probs = c.probs
        probs = prior_c.expand_as(post_probs)
        prior = Categorical(probs=probs)
    kl = kl_divergence(c, prior)
    return kl


@register_kl(RelaxedOneHotCategorical, Categorical)
def _kl_categorical_relaxed_onehot_categorical(p, q):
    t = p.probs * (p.logits - q.logits)
    t[(q.probs == 0).expand_as(t)] = torch.inf
    t[(p.probs == 0).expand_as(t)] = 0
    return t.sum(-1)


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
    def forward(self, batch: GMINIBATCH) -> FRES:
        """enocde"""

    @abstractmethod
    def step(self, batch: GMINIBATCH) -> Tuple[FRES, LOSS]:
        """encode and calculate losses"""


class AttVEncoder(BaseEncoder):
    def __init__(
        self,
        incs: Dict[str, int],
        outc: int,
        nbatch: int = 0,
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
        self.nbatch = nbatch
        self.unshared_mlps = nn.ModuleDict()
        for name, ninpt in incs.items():
            self.unshared_mlps[name] = VanillaMLP(
                inc=ninpt,
                outc=nlatent_middle,
                hiddens=hiddens_unshare,
                continue_cov_dims=[],
                discrete_cov_dims=[nbatch] if nbatch else [],
                act=act,
                bn=bn,
                dp=dp,
                last_act=True,
                last_bn=bn,
                last_dp=dp > 0,
            )

        fusion_inpt = {k: nlatent_middle for k in incs}
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

    def forward(self, batch: GMINIBATCH) -> FRES:
        embeds_dedicated = {}
        for k, mlpi in self.unshared_mlps.items():
            embeds_dedicated[k] = mlpi(
                batch["input"][k], [], [batch["blabel"]] if self.nbatch else []
            )
        h, att = self.fusion(embeds_dedicated, batch["mask"])
        _, z = self.mapper(h)
        return {"z": z, "att": att}

    def step(self, batch: GMINIBATCH) -> Tuple[FRES, LOSS]:
        z = self.forward(batch)
        loss_kl = calc_kl_normal(z["z"])
        loss_kl = self.reductor(loss_kl, dim=1).mean()
        return z, {"kl": loss_kl}

    def attention(self, batch: GMINIBATCH) -> T:
        embeds_dedicated = {}
        for k, mlpi in self.unshared_mlps.items():
            embeds_dedicated[k] = mlpi(
                batch["input"][k], [], [batch["blabel"]]
            )
        _, att = self.fusion(embeds_dedicated, batch["mask"])
        return att


class AttGMMVEncoder(AttVEncoder):
    def __init__(
        self,
        incs: Dict[str, int],
        zdim: int,
        cdim: int,
        udim: int,
        nbatch: int = 0,
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
        temperature: float = 1.0,
    ):
        super().__init__(
            incs=incs,
            outc=zdim,
            nbatch=nbatch,
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
        self.temp = temperature  # TODO: 变化的temperature
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
        self, batch: GMINIBATCH
    ) -> Dict[str, Union[Normal, Categorical, RelaxedOneHotCategorical]]:
        fres = super().forward(batch)
        _, c = self.q_c_z(fres["z"].mean, temperature=self.temp)
        fres["c"] = c
        return fres

    def step(
        self, batch: GMINIBATCH
    ) -> Tuple[
        Dict[str, Union[T, Normal, Categorical, RelaxedOneHotCategorical]],
        LOSS,
    ]:
        fres = super().forward(batch)
        z = fres["z"]
        zsample = z.rsample()
        # forward用mean，step用rsample
        _, c = self.q_c_z(zsample, temperature=self.temp)
        # fres = self.forward(batch)
        # z, c = fres["z"], fres["c"]
        # zsample = z.rsample()
        logq_z_x = z.log_prob(zsample)
        fres["zsample"] = zsample
        fres["c"] = c

        if self.ssl:
            sslabel = batch["sslabel"]
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

        return fres, losses


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

    def forward(self, batch: GMINIBATCH) -> Dict[str, Normal]:
        sidx, tidx, esgn, _, enorm = batch["normed_subgraph"]
        ptr = self.conv(self.vrepr, sidx, tidx, enorm, esgn)
        loc = self.loc(ptr)
        std = F.softplus(self.std_lin(ptr)) + 1e-7
        vdist = Normal(loc, std)
        return {"v": vdist, "vsample": vdist.rsample()}

    def step(self, batch: GMINIBATCH) -> Tuple[Dict[str, Normal], LOSS]:
        fres = self.forward(batch)
        loss_kl = calc_kl_normal(fres["v"])
        loss_kl = self.reductor(loss_kl, dim=1).mean()  # TODO: /z.shape[0]
        return fres, {"kl_graph": loss_kl}
