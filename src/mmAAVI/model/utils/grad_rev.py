import torch
from torch import nn
from torch.autograd import Function


class GradientReversalFunc(Function):
    @staticmethod
    def forward(ctx, x, alpha):
        alpha = torch.as_tensor(alpha).to(x)
        ctx.save_for_backward(x, alpha)
        return x

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = None
        _, alpha = ctx.saved_tensors
        if ctx.needs_input_grad[0]:
            grad_input = -alpha * grad_output
        return grad_input, None


class GradientReversal(nn.Module):
    def __init__(self, alpha):
        super().__init__()
        self.alpha = torch.tensor(alpha, requires_grad=False)

    def forward(self, x):
        return GradientReversalFunc.apply(x, self.alpha)


gradient_reversal = GradientReversalFunc.apply
