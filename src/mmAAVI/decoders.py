from abc import abstractmethod
from typing import (
    Mapping,
    Optional,
    Sequence,
    Tuple,
    Union,
    Dict,
    Any,
)

import torch
import torch.distributions as D
import torch.nn as nn

from .dataset import GMINIBATCH
from .model.block import (
    DistributionMLP,
    VanillaMLP,
    DistributionDotDecoder,
    DistributionMixtureDecoder,
)


T = torch.Tensor
FRES = Dict[str, Any]  # forward results
LOSS = Dict[str, T]


def split_v(v: T, feat_dims: Dict[str, int]) -> Dict[str, T]:
    # TODO: 这个分解顺序可能被保证吗？
    vs, start = {}, 0
    for k, ndim in feat_dims.items():
        vs[k] = v[start : (start + ndim), :]  # dim0才是feature的维度
        start += ndim
    return vs


class GraphDecoder:
    r"""
    Graph decoder
    """

    def forward(self, batch: GMINIBATCH, enc_res: FRES) -> FRES:
        v = enc_res["zsample"]
        sidx, tidx, esgn, _, _ = batch["normed_subgraph"]
        logits = esgn * (v[sidx] * v[tidx]).sum(dim=1)
        return {"g": D.Bernoulli(logits=logits)}

    def step(self, batch: GMINIBATCH, enc_res: FRES) -> Tuple[FRES, LOSS]:
        dec_res = self.forward(batch, enc_res)
        dist = dec_res["g"]
        ewt = batch["normed_subgraph"][-2]
        g_nll = -dist.log_prob(ewt)
        pos_mask = (ewt != 0).long()
        n_pos = pos_mask.sum().item()
        n_neg = pos_mask.numel() - n_pos
        g_nll_pn = torch.zeros(2, dtype=g_nll.dtype, device=g_nll.device)
        g_nll_pn.scatter_add_(0, pos_mask, g_nll)
        avgc = (n_pos > 0) + (n_neg > 0)
        g_nll = g_nll_pn[0] / max(n_neg, 1) + g_nll_pn[1] / max(n_pos, 1)
        g_nll = g_nll / avgc
        return dec_res, {"rec_graph": g_nll}


""" Multi-Omics Decoders """


class BaseMultiOmicsDecoder(nn.Module):
    def __init__(self, outcs: Dict[str, int], reduction: str = "sum") -> None:
        super().__init__()
        self.outcs = outcs
        if reduction == "sum":
            self.reductor = torch.sum
        elif reduction == "mean":
            self.reductor = torch.mean
        else:
            raise NotImplementedError

    @abstractmethod
    def forward(self, batch: GMINIBATCH, enc_res: FRES) -> FRES:
        """返回的是dict of distribution"""

    @abstractmethod
    def step(self, batch: GMINIBATCH, enc_res: FRES) -> Tuple[FRES, LOSS]:
        """返回的是dict of distribution和losses"""


class MLPMultiModalDecoder(BaseMultiOmicsDecoder):
    def __init__(
        self,
        inc: int,
        outcs: Dict[str, int],
        hiddens: Sequence[int],
        nbatch: int = 0,
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

        self.nbatch = nbatch
        self.d_names = distributions
        self.d_styles = distributions_style
        self.mlps = nn.ModuleDict()
        for k, outci in outcs.items():
            dnamei = distributions[k]
            dstylei = distributions_style[k]
            ccov_dims = [1] if dnamei == "nb" and dstylei == "batch" else []
            self.mlps[k] = DistributionMLP(
                inc=inc,
                outc=outci,
                hiddens=hiddens,
                continue_cov_dims=ccov_dims,
                discrete_cov_dims=[nbatch] if nbatch else [],
                act=act,
                bn=bn,
                dp=dp,
                last_act=False,
                last_bn=False,
                last_dp=False,
                distribution=dnamei,
                distribution_style=dstylei,
            )

    def forward(self, batch: GMINIBATCH, enc_res: FRES) -> FRES:
        res_h, res_d = {}, {}
        for k, mlpi in self.mlps.items():
            if self.d_names[k] == "nb" and self.d_styles[k] == "batch":
                library_size = batch["output"][k].sum(dim=1, keepdim=True)
                ccovs = [library_size]
            else:
                ccovs = []
            hi, di = mlpi(
                enc_res["zsample"],
                ccovs,
                [batch["blabel"]] if self.nbatch else [],
            )
            res_h[k] = hi
            res_d[k] = di
        return {"hidden": res_h, "dist": res_d}

    def step(self, batch: GMINIBATCH, enc_res: FRES) -> Tuple[FRES, LOSS]:
        fres = self.forward(batch, enc_res)
        losses = {}
        for k, disti in fres["dist"].items():
            nlli = -disti.log_prob(batch["output"][k])[batch["mask"][k] > 0.0]
            if nlli.size(0) > 0:
                lossi = self.reductor(nlli, dim=1).mean()
                losses[k] = lossi
        return fres, losses


class DotMultiModalDecoder(BaseMultiOmicsDecoder):
    def __init__(
        self,
        outcs: Dict[str, int],
        nbatch: int = 0,
        inpt: Optional[int] = None,
        hiddens: Optional[Sequence[int]] = None,
        act: str = "relu",
        bn: bool = True,
        dp: float = 0.0,
        reduction: str = "sum",
        distributions: Union[str, Mapping[str, str]] = "nb",
        distributions_style: Union[str, Mapping[str, str]] = "batch",
    ) -> None:
        """这个解码器可以在做dot之前先进行非线性映射"""
        super().__init__(outcs=outcs, reduction=reduction)
        if isinstance(distributions, str):
            distributions = {k: distributions for k in outcs.keys()}
        if isinstance(distributions_style, str):
            distributions_style = {
                k: distributions_style for k in outcs.keys()
            }

        self.nbatch = nbatch
        self.d_names = distributions
        self.d_styles = distributions_style
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
                outci,
                nbatch if nbatch else None,
                distributions[k],
                distributions_style[k],
            )

    def forward(self, batch: GMINIBATCH, enc_res: FRES) -> FRES:
        u = enc_res["zsample"]
        v = enc_res["vsample"]
        vs = split_v(v, self.outcs)
        if self.pre_mlp is not None:
            u = self.pre_mlp(u)
        res_d = {}
        for k, neti in self.nets.items():
            if self.d_names[k] == "nb" and self.d_styles[k] == "batch":
                ls = batch["output"][k].sum(dim=1, keepdim=True)
            else:
                ls = None
            di = neti(u, vs[k], batch["blabel"] if self.nbatch else None, ls)
            res_d[k] = di
        return {"dist": res_d}

    def step(self, batch: GMINIBATCH, enc_res: FRES) -> Tuple[FRES, LOSS]:
        fres = self.forward(batch, enc_res)
        losses = {}
        for k, disti in fres.items():
            nlli = -disti.log_prob(batch["output"][k])[batch["mask"][k] > 0.0]
            if nlli.size(0) > 0:
                lossi = self.reductor(nlli, dim=1).mean()
                losses[k] = lossi
        return fres, losses


class MixtureMultiModalDecoder(BaseMultiOmicsDecoder):
    def __init__(
        self,
        inc: int,
        outcs: Dict[str, int],
        hiddens: Sequence[int],
        nbatch: int = 0,
        act: str = "relu",
        bn: bool = True,
        dp: float = 0,
        weight_dot: float = 0.5,
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
        assert (weight_dot > 0.0) and (weight_dot < 1.0)

        super().__init__(outcs=outcs, reduction=reduction)

        if isinstance(distributions, str):
            distributions = {k: distributions for k in outcs.keys()}
        if isinstance(distributions_style, str):
            distributions_style = {
                k: distributions_style for k in outcs.keys()
            }

        self.weight_dot = weight_dot
        self.nbatch = nbatch
        self.d_names = distributions
        self.d_styles = distributions_style
        self.mlps = nn.ModuleDict()
        for k, outci in outcs.items():
            dnamei = distributions[k]
            dstylei = distributions_style[k]
            ccov_dims = [1] if dnamei == "nb" and dstylei == "batch" else []
            self.mlps[k] = DistributionMixtureDecoder(
                inc=inc,
                outc=outci,
                hiddens=hiddens,
                continue_cov_dims=ccov_dims,
                discrete_cov_dims=[nbatch] if nbatch else [],
                act=act,
                bn=bn,
                dp=dp,
                last_act=False,
                last_bn=False,
                last_dp=False,
                distribution=dnamei,
                distribution_style=dstylei,
            )

    def forward(self, batch: GMINIBATCH, enc_res: FRES) -> FRES:
        """这个模型仿效的是mlp decoder，所以不需要将library放在外面"""
        v = enc_res["vsample"]
        vs = split_v(v, self.outcs)

        res_h, res_d = {}, {}
        for k, mlpi in self.mlps.items():
            if self.d_names[k] == "nb" and self.d_styles[k] == "batch":
                library_size = batch["output"][k].sum(dim=1, keepdim=True)
                ccovs = [library_size]
            else:
                ccovs = []
            hi, di = mlpi(
                enc_res["zsample"],
                vs[k],
                ccovs,
                [batch["blabel"]] if self.nbatch else [],
                self.weight_dot,
                1 - self.weight_dot,
            )
            res_h[k] = hi
            res_d[k] = di
        return {"hidden": res_h, "dist": res_d}

    def step(self, batch: GMINIBATCH, enc_res: FRES) -> Tuple[FRES, LOSS]:
        fres = self.forward(batch, enc_res)
        losses = {}
        for k, disti in fres["dist"].items():
            nlli = -disti.log_prob(batch["output"][k])[batch["mask"][k] > 0.0]
            if nlli.size(0) > 0:
                lossi = self.reductor(nlli, dim=1).mean()
                losses[k] = lossi
        return fres, losses
