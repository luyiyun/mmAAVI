from collections import OrderedDict as odict
from typing import Dict, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from .weighter import Weighter

T = torch.Tensor


def onehot(arr: np.ndarray, n: int) -> np.ndarray:
    return np.eye(n)[arr.astype(int)]


def seq2ndarray(seq: Sequence) -> np.ndarray:
    if isinstance(seq, np.ndarray):
        return seq
    elif isinstance(seq, list):
        return np.array(seq)
    elif torch.is_tensor(seq):
        return seq.detach().cpu().numpy()
    else:
        raise NotImplementedError


def tensors_to_device(tensors, device: torch.device):
    if torch.is_tensor(tensors):
        return tensors.to(device)
    elif isinstance(tensors, str):
        return tensors
    elif isinstance(tensors, dict):
        return {k: tensors_to_device(v, device) for k, v in tensors.items()}
    elif isinstance(tensors, odict):
        return odict(
            [(k, tensors_to_device(v, device)) for k, v in tensors.items()]
        )
    elif isinstance(tensors, Sequence):
        return [tensors_to_device(t, device) for t in tensors]
    else:
        return tensors
        # raise ValueError("unknown type: %s" % (str(type(tensors))))


def concat_embeds(embeds) -> Tuple[T, Optional[T]]:
    if torch.is_tensor(embeds[0]):
        z = torch.cat(embeds, dim=0)
        clu_prob = None
    elif isinstance(embeds[0], (list, tuple)) and len(embeds[0]) >= 2:
        z, clu_prob = list(zip(*[output[:2] for output in embeds]))
        z = torch.cat(z, dim=0)
        clu_prob = torch.cat(clu_prob, dim=0)
    else:
        raise NotImplementedError
    return z, clu_prob


def append_hist_and_tensorboard(
    epoch: int,
    scores: Dict[str, float],
    hist: Dict[str, list],
    phase: str = "train",
    writer: Optional[SummaryWriter] = None,
) -> None:
    hist["epoch"].append(epoch)
    for k, v in scores.items():
        hist.setdefault(k, []).append(v)
    if writer is not None:
        for k, v in scores.items():
            writer.add_scalar("%s/%s" % (phase, k), v, epoch)
        writer.flush()


def sum_losses(epoch: int, losses: Dict[str, T], weighter: Weighter) -> T:
    loss = 0.0
    for k, v in losses.items():
        if k in weighter:
            loss += v * weighter.at(k, epoch)
        else:
            loss += v
    return loss
