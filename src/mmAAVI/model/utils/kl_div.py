from typing import Optional

import torch
import torch.distributions as D
from torch.distributions import Normal, Categorical, kl_divergence


T = torch.Tensor


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


@D.register_kl(D.RelaxedOneHotCategorical, D.Categorical)
def _kl_categorical_relaxed_onehot_categorical(p, q):
    t = p.probs * (p.logits - q.logits)
    t[(q.probs == 0).expand_as(t)] = torch.inf
    t[(p.probs == 0).expand_as(t)] = 0
    return t.sum(-1)
