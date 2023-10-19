import collections as ct
from abc import abstractmethod
from typing import (Literal, Mapping, Optional, OrderedDict, Sequence, Tuple,
                    Union)

import torch
import torch.distributions as D
import torch.nn as nn
import torch.nn.functional as F

from ..typehint import LOSS, T
from .block import (DistributionDotDecoder, DistributionMixtureDecoder,
                    DistributionMLP, VanillaMLP)

""" Graph Decoders """


class GraphDecoder:
    r"""
    Graph decoder
    """

    def forward(self, v: T, sidx: T, tidx: T, esgn: T) -> D.Bernoulli:
        logits = esgn * (v[sidx] * v[tidx]).sum(dim=1)
        return D.Bernoulli(logits=logits)

    def step(
        self, v: T, sidx: T, tidx: T, esgn: T, ewt: T
    ) -> Tuple[D.Bernoulli, LOSS]:
        dist = self.forward(v, sidx, tidx, esgn)
        g_nll = -dist.log_prob(ewt)
        pos_mask = (ewt != 0).long()
        n_pos = pos_mask.sum().item()
        n_neg = pos_mask.numel() - n_pos
        g_nll_pn = torch.zeros(2, dtype=g_nll.dtype, device=g_nll.device)
        g_nll_pn.scatter_add_(0, pos_mask, g_nll)
        avgc = (n_pos > 0) + (n_neg > 0)
        g_nll = g_nll_pn[0] / max(n_neg, 1) + g_nll_pn[1] / max(n_pos, 1)
        g_nll = g_nll / avgc
        return dist, {"rec_graph": g_nll}


""" Multi-Omics Decoders """


class BaseMultiOmicsDecoder(nn.Module):
    def __init__(
        self, outcs: OrderedDict[str, int], reduction: str = "sum"
    ) -> None:
        super().__init__()
        self.outcs = outcs
        if reduction == "sum":
            self.reductor = torch.sum
        elif reduction == "mean":
            self.reductor = torch.mean
        else:
            raise NotImplementedError

    @abstractmethod
    def forward(
        self,
        inp: T,
        continue_covs: list[T] = [],
        discrete_covs: list[T] = [],
    ) -> OrderedDict[str, D.Distribution]:
        """返回的是dict of distribution"""

    @abstractmethod
    def step(
        self,
        inp: T,
        oupt: OrderedDict[str, T],
        masks: Optional[OrderedDict[str, T]] = None,
        continue_covs: list[T] = [],
        discrete_covs: list[T] = [],
    ) -> Tuple[OrderedDict[str, D.Distribution], LOSS]:
        """返回的是dict of distribution和losses"""


class MLPMultiModalDecoder(BaseMultiOmicsDecoder):
    def __init__(
        self,
        inc: int,
        outcs: OrderedDict[str, int],
        hiddens: Sequence[int],
        continue_cov_dims: list[int] = [],
        discrete_cov_dims: list[int] = [],
        act: str = "relu",
        bn: bool = True,
        dp: float = 0,
        reduction: str = "sum",
        distributions: Union[str, Mapping[str, str]] = "nb",
        distributions_style: Union[str, Mapping[str, str]] = "batch",
    ) -> None:
        """
        如果是batch-style nb，则还需要输入library size到ccov中。
        因为可能不是所有的omics都使用nb，所以我们不能将那个1加入到这里的ccov中。
        所以默认这里的而ccov_dims是不包含这个1的，这个1将被自动包括进去
        同样，forward中输入的那个library size也不被包含在ccovs中，而是在kwargs
        """
        super().__init__(outcs=outcs, reduction=reduction)
        if isinstance(distributions, str):
            distributions = {k: distributions for k in outcs.keys()}
        if isinstance(distributions_style, str):
            distributions_style = {
                k: distributions_style for k in outcs.keys()
            }

        self.d_names = distributions
        self.d_styles = distributions_style
        self.mlps = nn.ModuleDict()
        for k, outci in outcs.items():
            dnamei = distributions[k]
            dstylei = distributions_style[k]
            if dnamei == "nb" and dstylei == "batch":
                ccov_dims = [1] + continue_cov_dims
            else:
                ccov_dims = continue_cov_dims
            self.mlps[k] = DistributionMLP(
                inc=inc,
                outc=outci,
                hiddens=hiddens,
                continue_cov_dims=ccov_dims,
                discrete_cov_dims=discrete_cov_dims,
                act=act,
                bn=bn,
                dp=dp,
                last_act=False,
                last_bn=False,
                last_dp=False,
                distribution=dnamei,
                distribution_style=dstylei,
            )

    def forward(
        self,
        inp: T,
        continue_covs: list[T] = [],
        discrete_covs: list[T] = [],
        outputs: Optional[OrderedDict[str, T]] = None,
        **kwargs
    ) -> Tuple[OrderedDict[str, T], OrderedDict[str, D.Distribution]]:
        res_h, res_d = ct.OrderedDict(), ct.OrderedDict()
        for k, mlpi in self.mlps.items():
            if self.d_names[k] == "nb" and self.d_styles[k] == "batch":
                library_size = outputs[k].sum(dim=1, keepdim=True)
                ccovs = [library_size] + continue_covs
            else:
                ccovs = continue_covs
            hi, di = mlpi(inp, ccovs, discrete_covs, **kwargs)
            res_h[k] = hi
            res_d[k] = di
        return res_h, res_d

    def step(
        self,
        inpt: T,
        oupt: OrderedDict[str, T],
        masks: Optional[OrderedDict[str, T]] = None,
        continue_covs: list[T] = [],
        discrete_covs: list[T] = [],
        prefix: str = "rec_",
    ) -> Tuple[OrderedDict[str, T], OrderedDict[str, D.Distribution], LOSS]:
        res_hs, rec_dists = self.forward(
            inpt, continue_covs, discrete_covs, oupt
        )
        losses = {}
        for k, disti in rec_dists.items():
            nlli = -disti.log_prob(oupt[k])[masks[k] > 0.0]
            if nlli.size(0) > 0:
                lossi = self.reductor(nlli, dim=1).mean()
                losses[prefix + k] = lossi
        return res_hs, rec_dists, losses


class DotMultiModalDecoder(BaseMultiOmicsDecoder):
    def __init__(
        self,
        outcs: OrderedDict[str, int],
        nbatches: Optional[int] = None,
        inpt: Optional[int] = None,
        hiddens: Optional[Sequence[int]] = None,
        act: str = "relu",
        bn: bool = True,
        dp: float = 0.0,
        reduction: str = "sum",
        distributions: Union[str, Mapping[str, str]] = "nb",
        distributions_style: Union[str, Mapping[str, str]] = "batch",
    ) -> None:
        super().__init__(outcs=outcs, reduction=reduction)
        if isinstance(distributions, str):
            distributions = {k: distributions for k in outcs.keys()}
        if isinstance(distributions_style, str):
            distributions_style = {
                k: distributions_style for k in outcs.keys()
            }

        if hiddens is None:
            self.pre_mlp = None
        else:
            assert (
                inpt is not None
            ), "inpt must be set when hiddens is not None"
            self.pre_mlp = VanillaMLP(
                inpt, inpt, hiddens, act=act, bn=bn, dp=dp
            )

        self.nets = nn.ModuleDict()
        for k, outci in outcs.items():
            self.nets[k] = DistributionDotDecoder(
                outci, nbatches, distributions[k], distributions_style[k]
            )

    def forward(
        self,
        u: T,
        vs: OrderedDict[str, T],
        b: Optional[T] = None,
        ls: Optional[OrderedDict[str, Optional[T]]] = None,
    ) -> OrderedDict[str, D.Distribution]:
        if self.pre_mlp is not None:
            u = self.pre_mlp(u)
        res_d = ct.OrderedDict()
        for k, neti in self.nets.items():
            di = neti(u, vs[k], b, None if ls is None else ls[k])
            res_d[k] = di
        return res_d

    def step(
        self,
        u: T,
        vs: OrderedDict[str, T],
        oupt: OrderedDict[str, T],
        masks: Optional[OrderedDict[str, T]] = None,
        b: Optional[T] = None,
        ls: Optional[OrderedDict[str, Optional[T]]] = None,
        prefix: str = "rec_",
    ) -> Tuple[OrderedDict[str, D.Distribution], LOSS]:
        rec_dists = self.forward(u, vs, b, ls)
        losses = {}
        for k, disti in rec_dists.items():
            nlli = -disti.log_prob(oupt[k])[masks[k] > 0.0]
            if nlli.size(0) > 0:
                lossi = self.reductor(nlli, dim=1).mean()
                losses[prefix + k] = lossi
        return rec_dists, losses


class MixtureMultiModalDecoder(BaseMultiOmicsDecoder):
    def __init__(
        self,
        inc: int,
        outcs: OrderedDict[str, int],
        hiddens: Sequence[int],
        continue_cov_dims: list[int] = [],
        discrete_cov_dims: list[int] = [],
        act: str = "relu",
        bn: bool = True,
        dp: float = 0,
        reduction: str = "sum",
        distributions: Union[str, Mapping[str, str]] = "nb",
        distributions_style: Union[str, Mapping[str, str]] = "batch",
    ) -> None:
        """
        如果是batch-style nb，则还需要输入library size到ccov中。
        因为可能不是所有的omics都使用nb，所以我们不能将那个1加入到这里的ccov中。
        所以默认这里的而ccov_dims是不包含这个1的，这个1将被自动包括进去
        同样，forward中输入的那个library size也不被包含在ccovs中，而是在kwargs
        """
        super().__init__(outcs=outcs, reduction=reduction)

        if isinstance(distributions, str):
            distributions = {k: distributions for k in outcs.keys()}
        if isinstance(distributions_style, str):
            distributions_style = {
                k: distributions_style for k in outcs.keys()
            }

        self.d_names = distributions
        self.d_styles = distributions_style
        self.mlps = nn.ModuleDict()
        for k, outci in outcs.items():
            dnamei = distributions[k]
            dstylei = distributions_style[k]
            if dnamei == "nb" and dstylei == "batch":
                ccov_dims = [1] + continue_cov_dims
            else:
                ccov_dims = continue_cov_dims
            self.mlps[k] = DistributionMixtureDecoder(
                inc=inc,
                outc=outci,
                hiddens=hiddens,
                continue_cov_dims=ccov_dims,
                discrete_cov_dims=discrete_cov_dims,
                act=act,
                bn=bn,
                dp=dp,
                last_act=False,
                last_bn=False,
                last_dp=False,
                distribution=dnamei,
                distribution_style=dstylei,
            )

    def forward(
        self,
        u: T,
        vs: OrderedDict[str, T],
        continue_covs: list[T] = [],
        discrete_covs: list[T] = [],
        outputs: Optional[OrderedDict[str, T]] = None,
        weight_dot: float = 0.5,
        weight_mlp: float = 0.5,
        library_size: Optional[OrderedDict[str, T]] = None,
        **kwargs
    ) -> Tuple[OrderedDict[str, T], OrderedDict[str, D.Distribution]]:
        """
        这个模型仿效的是mlp decoder，所以不需要将library放在外面
        """
        res_h, res_d = ct.OrderedDict(), ct.OrderedDict()
        for k, mlpi in self.mlps.items():
            if self.d_names[k] == "nb" and self.d_styles[k] == "batch":
                if library_size is not None:
                    library_size_k = library_size[k]
                else:
                    library_size_k = outputs[k].sum(dim=1, keepdim=True)
                ccovs = [library_size_k] + continue_covs
            else:
                ccovs = continue_covs
            hi, di = mlpi(
                u,
                vs[k],
                ccovs,
                discrete_covs,
                weight_dot,
                weight_mlp,
                **kwargs
            )
            res_h[k] = hi
            res_d[k] = di
        return res_h, res_d

    def step(
        self,
        u: T,
        vs: OrderedDict[str, T],
        oupt: OrderedDict[str, T],
        masks: Optional[OrderedDict[str, T]] = None,
        continue_covs: list[T] = [],
        discrete_covs: list[T] = [],
        weight_dot: float = 0.5,
        weight_mlp: float = 0.5,
        prefix: str = "rec_",
    ) -> Tuple[OrderedDict[str, T], OrderedDict[str, D.Distribution], LOSS]:
        _, rec_dists = self.forward(
            u, vs, continue_covs, discrete_covs, oupt, weight_dot, weight_mlp
        )
        losses = {}
        for k, disti in rec_dists.items():
            nlli = -disti.log_prob(oupt[k])[masks[k] > 0.0]
            if nlli.size(0) > 0:
                lossi = self.reductor(nlli, dim=1).mean()
                losses[prefix + k] = lossi
        return rec_dists, losses


""" Network Constraint """


class NetworkReconstractor(nn.Module):
    def __init__(
        self,
        inc: int,
        outcs: OrderedDict[str, int],
        style: Literal["lproj", "lproj_orth", "recon"] = "lproj",
    ) -> None:
        super().__init__()
        self.style = style
        self.maps = nn.ModuleDict()
        if style == "lproj":
            for k, dimi in outcs.items():
                self.maps[k] = nn.Linear(inc, dimi, bias=False)
        elif style == "lproj_orth":
            for k, dimi in outcs.items():
                self.maps[k] = nn.utils.parametrizations.orthogonal(
                    nn.Linear(inc, dimi, bias=False)
                )

    def step(
        self,
        varp: dict[str, tuple[T, T, T]],
        z: Optional[D.Normal] = None,
        rec_hs: Optional[OrderedDict[str, T]] = None,
    ) -> LOSS:
        if self.style == "recon":
            return self.calc_nct_losses(varp, rec_hs, "net_")
        elif self.style in ["lproj", "lproj_orth"]:
            recs = OrderedDict((k, m(z.mean)) for k, m in self.maps.items())
            return self.calc_nct_losses(varp, recs, "net_")

    @staticmethod
    def calc_nct_losses(
        varp: dict[tuple[str, str], tuple[T, T, T]],
        recs: OrderedDict[str, T],
        prefix: str = "net_",
    ) -> LOSS:
        losses = {}
        for (omic1, omic2), neti in varp.items():
            i, j, s, w = neti
            rec1, rec2 = recs[omic1], recs[omic2]
            if torch.is_tensor(rec1):
                v1, v2 = rec1[:, i], rec2[:, j]
            else:
                raise NotImplementedError

            # logits = (v1 * v2).sum(dim=0)
            logits = F.cosine_similarity(v1, v2, dim=0) * s  # 考虑到了sign
            nll = -D.Bernoulli(logits=logits).log_prob(w)
            # TODO: 这里简化了GLUE的loss function，但是应该有相同的效果
            losses[prefix + ("%s_%s" % (omic1, omic2))] = nll.mean()
        return losses
