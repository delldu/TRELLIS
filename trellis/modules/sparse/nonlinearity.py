import torch
import torch.nn as nn
from . import SparseTensor

__all__ = [
    'SparseSiLU',
    'SparseGELU',
]


class SparseSiLU(nn.SiLU):
    def forward(self, input: SparseTensor) -> SparseTensor:
        return input.replace(super().forward(input.feats), input.coords)

class SparseGELU(nn.GELU):
    def forward(self, input: SparseTensor) -> SparseTensor:
        return input.replace(super().forward(input.feats), input.coords)


