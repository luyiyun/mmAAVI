from typing import Union
import torch
import torch.nn.functional as F


T = torch.Tensor


def focal_loss(
    input: T, target: T, alpha: Union[T, float], gamma: Union[T, float]
) -> T:
    assert input.ndim == 2
    target = target.view(-1, 1)
    logpt = F.log_softmax(input, dim=1)
    logpt = logpt.gather(1, target)
    logpt = logpt.view(-1)
    pt = logpt.data.exp()

    if alpha is not None:
        # alpha = alpha.type_as(input.data)
        # at = alpha.gather(0, target.data.view(-1))
        logpt = logpt * alpha

    loss = -1 * ((1 - pt) ** gamma) * logpt
    return loss
