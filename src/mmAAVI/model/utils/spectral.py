import torch.nn as nn


def spectral_norm_linear(module: nn.Module):
    # if type(module) == nn.Linear:
    if isinstance(module, nn.Linear):
        nn.utils.parametrizations.spectral_norm(module)


def spectral_norm_module(module: nn.Module):
    module.apply(spectral_norm_linear)
    return module
