import torch
import torch.nn as nn
from . import SparseTensor
import pdb

__all__ = [
    'SparseLinear'
]

class SparseLinear(nn.Linear):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__(in_features, out_features, bias)

    def forward(self, input: SparseTensor) -> SparseTensor:
        output_features = torch.matmul(input.feats, self.weight.t())
        output_features = output_features + self.bias        
        return input.replace(output_features, input.coords)
