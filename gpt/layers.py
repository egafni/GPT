import torch
from torch import nn



class FlattenConsecutive(nn.Module):
    def __init__(self, n):
        super().__init__()
        self.n = n

    def __call__(self, x):
        B, T, C = x.shape
        x = x.contiguous().view(B, T // self.n, C * self.n)
        if x.shape[1] == 1:
            x = x.squeeze(1)
        self.out = x
        return self.out


class Permute(nn.Module):
    def __init__(self, dims):
        super().__init__()
        self.dims = dims
        # self._parameters = None

    def forward(self, x):
        return torch.permute(x, self.dims)
