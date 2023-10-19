from typing import Sequence, Optional, Literal, Tuple

import torch
# import torch.nn as nn
import torch.nn.functional as F

from .block import VanillaMLP
from .utils import spectral_norm_module, gradient_reversal, focal_loss
from ..typehint import T, LOSS


class Discriminator(VanillaMLP):

    def __init__(
        self, inc: int, outc: int, hiddens: Sequence[int],
        continue_cov_dims: list[int] = ..., discrete_cov_dims: list[int] = ...,
        act: str = "relu", bn: bool = True, dp: float = 0,
        criterion: Literal["ce", "bce", "focal"] = "ce",
        spectral_norm: bool = False, gradient_reversal: bool = False,
        # label_smooth: float = 0.0
    ) -> None:
        assert criterion in ["ce", "bce", "focal"]
        if criterion == "bce":
            assert outc == 1
        else:
            assert outc > 1

        super().__init__(
            inc, outc, hiddens, continue_cov_dims, discrete_cov_dims,
            act, bn, dp, False, False, False
        )
        self.sn = spectral_norm
        self.gr = gradient_reversal
        self.criterion = criterion
        if self.sn:
            spectral_norm_module(self)

    def forward(
        self, inpt: T,
        continue_covs: list[T] = [],
        discrete_covs: list[T] = [],
        **kwargs
    ) -> T:
        if len(continue_covs) > 0 or len(discrete_covs) > 0:
            inpt = [inpt]
            if continue_covs is not None:
                inpt.extend(list(continue_covs))
            if discrete_covs is not None:
                for i, t in enumerate(discrete_covs):
                    inpt.append(F.one_hot(t, self.dcov_dims[i]).to(inpt[0]))
            inpt = torch.cat(inpt, dim=-1)

        # NOTE: 保证如果covs也有梯度流动，接受的梯度是反转后的梯度
        if self.gr:
            inpt = gradient_reversal(inpt, kwargs["alpha"])

        return self.net(inpt)

    def step(
        self, inpt: T, label: T,
        continue_covs: list[T] = [],
        discrete_covs: list[T] = [],
        weights: Optional[T] = None,
        postfix: str = "",
        **kwargs
    ) -> Tuple[T, LOSS]:
        logit = self.forward(inpt, continue_covs, discrete_covs, **kwargs)
        if self.criterion == "ce":
            loss = F.cross_entropy(
                logit, label,
                label_smoothing=kwargs.get("label_smooth", 0.0),
                reduction="none"
            )
        elif self.criterion == "bce":
            loss = F.binary_cross_entropy_with_logits(
                logit.squeeze(), label, reduction="none"
            )
        elif self.criterion == "focal":
            loss = focal_loss(
                logit, label, kwargs["focal_alpha"], kwargs["focal_gamma"]
            )

        if weights is not None:
            loss = loss * weights

        return logit, {"disc" + postfix: torch.mean(loss)}
