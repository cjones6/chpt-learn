import torch
import torch.nn as nn

import src.default_params as defaults


class SumSigmoids(nn.Module):
    def __init__(self, m=3):
        super().__init__()
        self.thetas = torch.nn.Parameter(torch.Tensor(m, 3))

    def forward(self, x):
        return torch.sum(self.thetas[:, 2] * torch.exp(-0.5 * (torch.mm(x, self.thetas[:, 0].unsqueeze(0)) +
                                                               self.thetas[:, 1]) ** 2), 1, keepdim=True)


def init_normal(m, std=1):
    """
    Initialize the weights of a layer m with draws from the random normal distribution with mean 0 and the given
    standard deviation.
    """
    try:
        m.thetas.data.normal_(0.0, std)
    except:
        pass
