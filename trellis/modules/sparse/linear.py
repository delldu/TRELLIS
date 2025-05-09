import torch
import torch.nn as nn
from . import SparseTensor
import pdb

__all__ = [
    'SparseLinear'
]


class SparseLinear(nn.Linear):
    def __init__(self, in_features, out_features, bias=True):
        super(SparseLinear, self).__init__(in_features, out_features, bias)

    def forward(self, input: SparseTensor) -> SparseTensor:
        # pdb.set_trace()
        return input.replace(super().forward(input.feats))
