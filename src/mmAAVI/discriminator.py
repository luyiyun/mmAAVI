from typing import Sequence, Optional, Literal, Tuple, Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from .model.block import VanillaMLP
from .model.utils import spectral_norm_module, gradient_reversal, focal_loss
from .dataset import GMINIBATCH


T = torch.Tensor
FRES = Dict[str, Any]  # forward results
LOSS = Dict[str, T]


class Discriminator(nn.Module):
    def __init__(
        self,
        inc: int,
        outc: int,
        hiddens: Sequence[int],
        nclusters: Optional[int] = None,
        disc_on_mean: bool = True,
        disc_condi_train: Optional[Literal["hard", "soft"]] = None,
        act: str = "relu",
        bn: bool = True,
        dp: float = 0,
        criterion: Literal["ce", "bce", "focal"] = "ce",
        spectral_norm: bool = False,
        gradient_reversal: bool = False,
        gradient_alpha: float = 1.0,
        label_smooth: float = 0.0,
        focal_alpha: float = 2.0,
        focal_gamma: float = 1.0,
    ) -> None:
        assert criterion in ["ce", "bce", "focal"]
        if criterion == "bce":
            assert outc == 1
        else:
            assert outc > 1
        if disc_condi_train is not None:
            assert disc_condi_train in ["soft", "hard"]

        super().__init__()

        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma
        self.label_smooth = label_smooth
        self.grad_alpha = gradient_alpha
        self.disc_condi_train = disc_condi_train
        self.disc_on_mean = disc_on_mean
        ccdims = [nclusters] if disc_condi_train == "soft" else []
        dcdims = [nclusters] if disc_condi_train == "hard" else []

        self.net = VanillaMLP(
            inc,
            outc,
            hiddens,
            ccdims,
            dcdims,
            act,
            bn,
            dp,
            False,
            False,
            False,
            False,
        )

        self.sn = spectral_norm
        self.gr = gradient_reversal
        self.criterion = criterion
        if self.sn:
            spectral_norm_module(self.net)

    def forward(self, batch: GMINIBATCH, enc_res: FRES, dec_res: FRES) -> FRES:
        if self.disc_condi_train is not None:
            cprobs = enc_res["c"].probs.detach()
            if self.disc_condi_train == "soft":
                ccovs, dcovs = [cprobs], []
            else:
                ccovs, dcovs = [], [cprobs.argmax(dim=1)]
        else:
            ccovs, dcovs = [], []
        # if len(continue_covs) > 0 or len(discrete_covs) > 0:
        #     inpt = [inpt]
        #     if continue_covs is not None:
        #         inpt.extend(list(continue_covs))
        #     if discrete_covs is not None:
        #         for i, t in enumerate(discrete_covs):
        #             inpt.append(F.one_hot(t, self.dcov_dims[i]).to(inpt[0]))
        #     inpt = torch.cat(inpt, dim=-1)

        # NOTE: Ensure that if covs also has gradient flow, the received
        #   gradient is the reversed gradient.
        # Directly cut off the gradient here to ensure there is
        #   no gradient flow.
        # TODO: Therefore, this implementation is actually different from the
        #   previous one, mainly in the position of the gradient reversal.
        inpt = enc_res["z"].mean if self.disc_on_mean else enc_res["zsample"]
        if self.gr:
            inpt = gradient_reversal(inpt, self.grad_alpha)

        return {"logit": self.net(inpt, ccovs, dcovs)}

    def step(
        self, batch: GMINIBATCH, enc_res: FRES, dec_res: FRES
    ) -> Tuple[FRES, LOSS]:
        fres = self.forward(batch, enc_res, dec_res)
        logit = fres["logit"]
        label = batch["dlabel"]
        if self.criterion == "ce":
            loss = F.cross_entropy(
                logit,
                label,
                label_smoothing=self.label_smooth,
                reduction="none",
            )
        elif self.criterion == "bce":
            loss = F.binary_cross_entropy_with_logits(
                logit.squeeze(), label, reduction="none"
            )
        elif self.criterion == "focal":
            loss = focal_loss(logit, label, self.focal_alpha, self.focal_gamma)

        # if weights is not None:
        #     loss = loss * weights

        return fres, {"disc": torch.mean(loss)}
