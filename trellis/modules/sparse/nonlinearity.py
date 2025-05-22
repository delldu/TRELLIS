import torch
import torch.nn as nn
from . import SparseTensor
import torch.nn.functional as F

__all__ = [
    'SparseSiLU',
    'SparseGELU',
]

class SparseSiLU(nn.SiLU):
    def forward(self, input: SparseTensor) -> SparseTensor:
        # return input.replace(super().forward(input.feats))
        x = F.silu(input.feats, inplace=self.inplace)
        return input.replace(x)

class SparseGELU(nn.GELU):
    def forward(self, input: SparseTensor) -> SparseTensor:
        # return input.replace(super().forward(input.feats))
        x = F.gelu(input.feats, approximate=self.approximate)
        return input.replace(x)


