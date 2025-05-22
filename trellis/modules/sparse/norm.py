import torch
import torch.nn as nn
import torch.nn.functional as F

from . import SparseTensor
import pdb

__all__ = [
    'SparseGroupNorm',
    'SparseLayerNorm',
]

class SparseGroupNorm(nn.GroupNorm):
    def __init__(self, num_groups, num_channels, eps=1e-5, affine=True):
        super().__init__(num_groups, num_channels, eps, affine)

    def forward(self, input: SparseTensor) -> SparseTensor:
        bfeats = input.feats
        bfeats = bfeats.permute(1, 0).reshape(1, input.shape[1], -1)
        # bfeats = super().forward(bfeats)
        bfeats = F.group_norm(bfeats, self.num_groups, self.weight, self.bias, self.eps)
        bfeats = bfeats.reshape(input.shape[1], -1).permute(1, 0)
        return input.replace(bfeats)


class SparseLayerNorm(nn.LayerNorm):
    def __init__(self, normalized_shape, eps=1e-5, elementwise_affine=True):
        super().__init__(normalized_shape, eps, elementwise_affine)

    def forward(self, input: SparseTensor) -> SparseTensor:
        bfeats = input.feats
        bfeats = bfeats.permute(1, 0).reshape(1, input.shape[1], -1)
        # bfeats = super().forward(bfeats)
        bfeats = F.layer_norm(bfeats, self.normalized_shape, self.weight, self.bias, self.eps)
        bfeats = bfeats.reshape(input.shape[1], -1).permute(1, 0)
        return input.replace(bfeats)


