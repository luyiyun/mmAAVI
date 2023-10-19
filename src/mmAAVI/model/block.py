from functools import partial
from typing import Literal, Optional, OrderedDict, Sequence, Tuple, Union

import torch
import torch.distributions as D
import torch.nn as nn
import torch.nn.functional as F

from ..constant import EPS
from .utils import ZILN

T = torch.Tensor
dist_names = Literal[
    "normal", "nb", "zinb", "categorical", "categorical_gumbel"
]
dist_style = Literal["sample", "same", "1", "batch"]
# sample代表variance由样本决定； same代表variance在各样本间相同；
# 1代表variance=1；
_ACTS = {
    "relu": nn.ReLU,
    "lrelu": partial(nn.LeakyReLU, negative_slope=0.2),
    "selu": nn.SELU,
    "elu": nn.ELU,
    "gelu": nn.GELU,
}


""" layers """


class GraphConv(nn.Module):
    r"""
    Graph convolution (propagation only)
    """

    def forward(self, input: T, sidx: T, tidx: T, enorm: T, esgn: T) -> T:
        r"""
        Forward propagation

        Parameters
        ----------
        input
            Input data (:math:`n_{vertices} \times n_{features}`)
        eidx
            Vertex indices of edges (:math:`2 \times n_{edges}`)
        enorm
            Normalized weight of edges (:math:`n_{edges}`)
        esgn
            Sign of edges (:math:`n_{edges}`)

        Returns
        -------
        result
            Graph convolution result (:math:`n_{vertices} \times n_{features}`)
        """
        message = input[sidx] * (esgn * enorm).unsqueeze(
            1
        )  # n_edges * n_features
        res = torch.zeros_like(input)
        tidx = tidx.unsqueeze(1).expand_as(message)  # n_edges * n_features
        res.scatter_add_(0, tidx, message)
        return res


class BN(nn.BatchNorm1d):
    def forward(self, x):
        if x.ndim == 2:
            return super().forward(x)
        elif x.ndim == 3:
            # 如果是=3个维度，则需要先将后面的维度翻转到前面来，做完BN后再翻转回去
            return super().forward(x.transpose(1, 2)).transpose(1, 2)
        else:
            raise ValueError("BN just for dim = 2 or 3")


""" MLP blocks """


class VanillaMLP(nn.Module):
    def __init__(
        self,
        inc: int,
        outc: int,
        hiddens: Sequence[int],
        continue_cov_dims: list[int] = [],
        discrete_cov_dims: list[int] = [],
        act: str = "relu",
        bn: bool = True,
        dp: float = 0.0,
        last_act: bool = False,
        last_bn: bool = False,
        last_dp: bool = False,
        act_before_bn: bool = False,
    ) -> None:
        assert isinstance(continue_cov_dims, list)
        assert isinstance(discrete_cov_dims, list)
        super().__init__()
        self.ccov_dims = continue_cov_dims
        self.dcov_dims = discrete_cov_dims

        inc = inc + sum(continue_cov_dims) + sum(discrete_cov_dims)
        hiddens = list(hiddens)
        layers = nn.ModuleList()
        act_func = _ACTS[act]
        layers = []
        if hiddens:
            for i, o in zip([inc] + hiddens[:-1], hiddens):
                layers.append(nn.Linear(i, o))
                if act_before_bn:
                    layers.append(act_func())
                    if bn:
                        layers.append(BN(o))
                else:
                    if bn:
                        layers.append(BN(o))
                    layers.append(act_func())
                if dp:
                    layers.append(nn.Dropout(dp))
            last_i = hiddens[-1]
        else:
            last_i = inc

        # 构造最后一层
        layers.append(nn.Linear(last_i, outc))
        if act_before_bn:
            if last_act:
                layers.append(act_func())
            if last_bn:
                layers.append(BN(outc))
        else:
            if last_bn:
                layers.append(BN(outc))
            if last_act:
                layers.append(act_func())
        if last_dp:
            layers.append(nn.Dropout(dp))

        self.net = nn.Sequential(*layers)

    def forward(
        self,
        inpt: T,
        continue_covs: list[T] = [],
        discrete_covs: list[T] = [],
    ) -> T:
        if len(continue_covs) == 0 and len(discrete_covs) == 0:
            return self.net(inpt)

        inpt = [inpt]
        if continue_covs is not None:
            inpt.extend(list(continue_covs))
        if discrete_covs is not None:
            for i, t in enumerate(discrete_covs):
                inpt.append(F.one_hot(t, self.dcov_dims[i]).to(inpt[0]))
        inpt = torch.cat(inpt, dim=-1)
        return self.net(inpt)


class DistributionTransformer(nn.Module):

    """将输出转换为distribution"""

    @staticmethod
    def check_params(
        continue_cov_dims: list[int],
        discrete_cov_dims: list[int],
        distribution: dist_names,
        distribution_style: dist_style,
    ) -> Tuple[Optional[int], Optional[int], list[int], list[int]]:
        assert distribution in [
            "normal",
            "nb",
            "ziln",
            "categorical",
            "categorical_gumbel",
        ]
        assert distribution_style in ["sample", "batch", "same", "1"]
        if distribution_style == "batch":
            assert (
                len(discrete_cov_dims) > 0
            ), "the first element of discrete_covs must be the batch dims"
        if distribution == "nb":
            assert (
                distribution_style != "1"
            ), "negative binomial doesn't has 1-style"
        if distribution == "ziln":
            assert (
                distribution_style == "batch"
            ), "zero-inflated log normal only has batch-style implementation"
        if distribution == "nb" and distribution_style == "batch":
            assert continue_cov_dims[0] == 1, (
                "the first of continue_cov_dims must be dim of library size "
                "(equal 1) when using batch-style negative binomial"
            )

        if distribution.startswith("categorical"):
            return None, None, continue_cov_dims, discrete_cov_dims

        nbatches, lsize = None, None
        if distribution_style == "batch":
            # 此时，第一个离散协变量就是batch index
            nbatches, *discrete_cov_dims = discrete_cov_dims
            if distribution == "nb":
                # 此时，第一个连续协变量就是library size
                lsize, *continue_cov_dims = continue_cov_dims

        return nbatches, lsize, continue_cov_dims, discrete_cov_dims

    def __init__(
        self,
        outc: int,
        nbatches: Optional[int],
        lsize: Optional[int],
        continue_cov_dims: list[int],
        discrete_cov_dims: list[int],
        distribution: dist_names,
        distribution_style: dist_style,
    ) -> None:
        super().__init__()

        self.outc = outc
        self.nbatches = nbatches
        self.lsize = lsize
        self.dcov_dims = discrete_cov_dims
        self.ccov_dims = continue_cov_dims
        self.d_name = distribution
        self.d_style = distribution_style

        """ 在network中创建一些专用的parameters """
        if self.d_name.startswith("categorical"):
            return
        if self.d_style == "batch":
            self.scale_lin = nn.Parameter(
                torch.zeros(self.nbatches, self.outc)
            )
            self.bias = nn.Parameter(torch.zeros(self.nbatches, self.outc))
            self.std_lin = nn.Parameter(torch.zeros(self.nbatches, self.outc))
            if self.d_name == "ziln":
                self.zi_logits = nn.Parameter(
                    torch.zeros(self.nbatches, self.outc)
                )
        elif self.d_style == "same":
            self.sigma_lin = nn.Parameter(torch.zeros(self.outc))

    def before_forward(
        self, continue_covs: list[T], discrete_covs: list[T]
    ) -> Tuple[Optional[T], Optional[T], list[T], list[T]]:
        if self.d_name.startswith("categorical"):
            return None, None, continue_covs, discrete_covs
        b, ls = None, None
        if self.d_style == "batch":
            b, *discrete_covs = discrete_covs
            if self.d_name == "nb":
                ls, *continue_covs = continue_covs
        return b, ls, continue_covs, discrete_covs

    def after_forward(
        self, h: T, b: Optional[T] = None, ls: Optional[T] = None, **kwargs
    ) -> Tuple[T, D.Distribution]:
        if self.d_name == "categorical":
            return h, D.Categorical(logits=h)
        elif self.d_name == "categorical_gumbel":
            return h, D.RelaxedOneHotCategorical(
                kwargs["temperature"], logits=h
            )

        if self.d_style == "sample":
            mu, sigma = h[..., : self.outc], h[..., self.outc:]
        elif self.d_style == "same":
            mu, sigma = h, self.sigma_lin
        elif self.d_style == "batch":
            scale = F.softplus(self.scale_lin[b])
            mu = scale * h + self.bias[b]
            sigma = self.std_lin[b]
        else:
            mu = h

        if self.d_name == "normal":
            if self.d_style == "1":
                return mu, D.Normal(mu, 1.0)
            else:
                std = F.softplus(sigma) + EPS
                return mu, D.Normal(mu, std)
        elif self.d_name == "nb":
            if self.d_style == "same":
                return mu, D.NegativeBinomial(sigma.exp(), logits=mu - sigma)
            elif self.d_style == "batch":
                mu = F.softmax(mu, dim=-1) * ls
                return mu, D.NegativeBinomial(
                    sigma.exp(), logits=(mu + EPS).log() - sigma
                )
            elif self.d_style == "sample":
                return mu, D.NegativeBinomial(sigma.exp(), logits=mu - sigma)
            else:
                raise NotImplementedError
        elif self.d_name == "ziln":
            std = F.softplus(sigma) + EPS
            return mu, ZILN(self.zi_logits[b].expand_as(mu), mu, std)
        else:
            raise NotImplementedError(self.d_name)


class DistributionMLP(VanillaMLP):

    """
    如果分布需要batch和library_size的额外输入，则其分别是放在discrete_covs和
    continue_covs中送入模型的
    """

    def __init__(
        self,
        inc: int,
        outc: int,
        hiddens: Sequence[int],
        continue_cov_dims: list[int] = [],
        discrete_cov_dims: list[int] = [],
        act: str = "relu",
        bn: bool = True,
        dp: float = 0,
        last_act: bool = False,
        last_bn: bool = False,
        last_dp: bool = False,
        distribution: Optional[dist_names] = "normal",
        distribution_style: dist_style = "sample",
    ) -> None:
        if distribution is None:
            self.d_transfer = None
            super().__init__(
                inc,
                outc,
                hiddens,
                continue_cov_dims,
                discrete_cov_dims,
                act,
                bn,
                dp,
                last_act,
                last_bn,
                last_dp,
            )
            return

        (
            nbatches,
            lsize,
            ccov_dims,
            dcov_dims,
        ) = DistributionTransformer.check_params(
            continue_cov_dims,
            discrete_cov_dims,
            distribution,
            distribution_style,
        )

        if distribution_style == "sample" and not distribution.startswith(
            "categorical"
        ):
            self.out_mul = 2
        else:
            self.out_mul = 1

        super().__init__(
            inc,
            outc * self.out_mul,
            hiddens,
            ccov_dims,
            dcov_dims,
            act,
            bn,
            dp,
            last_act,
            last_bn,
            last_dp,
        )

        self.d_transfer = DistributionTransformer(
            outc,
            nbatches,
            lsize,
            continue_cov_dims,
            discrete_cov_dims,
            distribution,
            distribution_style,
        )

    def forward(
        self,
        inpt: T,
        continue_covs: list[T] = [],
        discrete_covs: list[T] = [],
        **kwargs
    ) -> Tuple[T, D.Distribution]:
        if self.d_transfer is None:
            return super().forward(inpt, continue_covs, discrete_covs)
        b, l, continue_covs, discrete_covs = self.d_transfer.before_forward(
            continue_covs, discrete_covs
        )
        h = super().forward(inpt, continue_covs, discrete_covs)
        return self.d_transfer.after_forward(h, b, l, **kwargs)


""" fusion blocks """


class ConcatFusion(DistributionMLP):
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
        last_act: bool = False,
        last_bn: bool = False,
        last_dp: bool = False,
        distribution: Optional[dist_names] = "normal",
        distribution_style: dist_style = "sample",
    ) -> None:
        self.incs = incs
        inc = sum(incs.values())
        super().__init__(
            inc,
            outc,
            hiddens,
            continue_cov_dims,
            discrete_cov_dims,
            act,
            bn,
            dp,
            last_act,
            last_bn,
            last_dp,
            distribution,
            distribution_style,
        )

    def forward(
        self,
        inpts: OrderedDict[str, T],
        masks: Optional[OrderedDict[str, T]] = None,
        continue_covs: list[T] = [],
        discrete_covs: list[T] = [],
        **kwargs
    ) -> Union[T, D.Distribution]:
        if masks is not None:
            inpts = [inpts[n] * masks[n].unsqueeze(-1) for n in self.incs]
        inpts = torch.cat(inpts, dim=-1)
        return super().forward(inpts, continue_covs, discrete_covs, **kwargs)


class GatedAttentionFusion(nn.Module):
    def __init__(
        self,
        incs: OrderedDict[str, int],
        omic_embed: bool = False,
        omic_embed_train: bool = False,
    ):
        assert len(set(incs.values())) == 1, "GatedAttention需要各组学输入相同"
        super().__init__()
        self.incs = incs
        self.omic_embed = omic_embed
        inpt = list(incs.values())[0]

        if omic_embed and omic_embed_train:
            self.omic_vec = nn.Parameter(torch.zeros(self.n_omics, inpt))
        elif omic_embed:
            self.register_buffer("omic_vec", torch.zeros(self.n_omics, inpt))
        if omic_embed:
            nn.init.trunc_normal_(self.omic_vec)

        self.att1 = nn.Sequential(nn.Linear(inpt, 1), nn.Sigmoid())
        self.att2 = nn.Sequential(nn.Linear(inpt, 1), nn.Tanh())

    def forward(
        self,
        h_dict: OrderedDict[str, T],
        ex_dict: Optional[OrderedDict[str, T]] = None,
    ) -> Tuple[T, T]:
        h = torch.stack([h_dict[n] for n in self.incs], dim=1)
        if self.omic_embed:
            h = h + self.omic_vec
        att = self.att1(h) * self.att2(h)
        if ex_dict is not None:
            mask = torch.stack([ex_dict[n] for n in self.incs], dim=1)
            mask = mask.unsqueeze(-1)
            att[mask == 0.0] -= torch.inf  # 不能使用*，因为0*inf是nan
        att = torch.softmax(att, dim=1)
        h = (att * h).sum(dim=1)
        return h, att


""" dot decoders (use for merge graph and latent information) """


class DistributionDotDecoder(nn.Module):
    def __init__(
        self,
        outc: int,
        nbatches: Optional[int] = None,
        distribution: Optional[dist_names] = "normal",
        distribution_style: dist_style = "batch",
    ) -> None:
        super().__init__()
        assert (
            distribution_style != "sample"
        ), "sample-style is not allowed in DotDecoder"
        if (nbatches is None and distribution_style == "batch") or (
            nbatches is not None and distribution_style != "batch"
        ):
            raise ValueError("nbatches must be given when use batch-style")

        self.dist_name = distribution
        self.dist_style = distribution_style
        if distribution is None:
            return

        ccov_dims, dcov_dims = [], []
        if distribution_style == "batch":
            dcov_dims.append(nbatches)
            if distribution == "nb":
                ccov_dims.append(1)

        self.d_transfer = DistributionTransformer(
            outc, nbatches, 1, [], [], distribution, distribution_style
        )

    def forward(
        self, u: T, v: T, b: Optional[T] = None, ls: Optional[T] = None
    ) -> D.Distribution:
        h = u @ v.t()
        if self.dist_name is None:
            return h
        return self.d_transfer.after_forward(h, b, ls)[1]


""" mlp+dot decoders """


class DistributionMixtureDecoder(DistributionMLP):
    def __init__(
        self,
        inc: int,
        outc: int,
        hiddens: Sequence[int],
        continue_cov_dims: list[int] = [],
        discrete_cov_dims: list[int] = [],
        act: str = "relu",
        bn: bool = True,
        dp: float = 0,
        last_act: bool = False,
        last_bn: bool = False,
        last_dp: bool = False,
        distribution: Optional[dist_names] = "normal",
        distribution_style: dist_style = "sample",
    ) -> None:
        assert (
            distribution_style != "sample"
        ), "sample-style is not allowed in MixtureDecoder"

        super().__init__(
            inc,
            outc,
            hiddens,
            continue_cov_dims,
            discrete_cov_dims,
            act,
            bn,
            dp,
            last_act,
            last_bn,
            last_dp,
            distribution,
            distribution_style,
        )

    def forward(
        self,
        u: T,
        v: T,
        continue_covs: list[T] = [],
        discrete_covs: list[T] = [],
        weight_dot: float = 0.5,
        weight_mlp: float = 0.5,
        **kwargs
    ) -> Tuple[T, D.Distribution]:
        out_dot = u @ v.t()

        if self.d_transfer is None:
            # 实际上调用的是VanillaMLP的forward
            out_mlp = super(DistributionMLP, self).forward(
                u, continue_covs, discrete_covs
            )
            out = weight_dot * out_dot + weight_mlp * out_mlp
            return out

        b, l, continue_covs, discrete_covs = self.d_transfer.before_forward(
            continue_covs, discrete_covs
        )
        out_mlp = super(DistributionMLP, self).forward(
            u, continue_covs, discrete_covs
        )
        out = weight_dot * out_dot + weight_mlp * out_mlp
        return self.d_transfer.after_forward(out, b, l, **kwargs)
