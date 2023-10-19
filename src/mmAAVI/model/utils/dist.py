import torch
import torch.nn.functional as F
from torch.distributions import LogNormal

from ...constant import EPS


class ZILN(LogNormal):

    r"""
    Zero-inflated log-normal distribution with subsetting support

    Parameters
    ----------
    zi_logits
        Zero-inflation logits
    loc
        Location of the log-normal distribution
    scale
        Scale of the log-normal distribution
    """

    def __init__(
        self, zi_logits: torch.Tensor, loc: torch.Tensor, scale: torch.Tensor
    ) -> None:
        super().__init__(loc, scale)
        self.zi_logits = zi_logits

    def log_prob(self, value: torch.Tensor) -> torch.Tensor:
        zi_log_prob = torch.empty_like(value)
        z_mask = value.abs() < EPS
        z_zi_logits, nz_zi_logits = self.zi_logits[z_mask], self.zi_logits[~z_mask]
        zi_log_prob[z_mask] = z_zi_logits - F.softplus(z_zi_logits)
        zi_log_prob[~z_mask] = LogNormal(
            self.loc[~z_mask], self.scale[~z_mask]
        ).log_prob(value[~z_mask]) - F.softplus(nz_zi_logits)
        return zi_log_prob
