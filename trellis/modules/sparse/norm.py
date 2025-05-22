import torch
import torch.nn as nn
from . import SparseTensor
from . import DEBUG
import pdb

__all__ = [
    'SparseGroupNorm',
    'SparseLayerNorm',
    'SparseGroupNorm32',
    'SparseLayerNorm32',
]

class SparseGroupNorm(nn.GroupNorm):
    def __init__(self, num_groups, num_channels, eps=1e-5, affine=True):
        super().__init__(num_groups, num_channels, eps, affine)

    def forward(self, input: SparseTensor) -> SparseTensor:
        nfeats = torch.zeros_like(input.feats)
        assert input.shape[0] == 1

        # for k in range(input.shape[0]):
        #     bfeats = input.feats
        #     bfeats = bfeats.permute(1, 0).reshape(1, input.shape[1], -1)
        #     bfeats = super().forward(bfeats)
        #     bfeats = bfeats.reshape(input.shape[1], -1).permute(1, 0)
        #     nfeats = bfeats # xxxx_3333

        # assert (nfeats - bfeats).abs().max() < 0.01
        # return input.replace(nfeats)

        bfeats = input.feats
        bfeats = bfeats.permute(1, 0).reshape(1, input.shape[1], -1)
        bfeats = super().forward(bfeats)
        bfeats = bfeats.reshape(input.shape[1], -1).permute(1, 0)
        return input.replace(bfeats)


class SparseLayerNorm(nn.LayerNorm):
    def __init__(self, normalized_shape, eps=1e-5, elementwise_affine=True):
        super().__init__(normalized_shape, eps, elementwise_affine)

    def forward(self, input: SparseTensor) -> SparseTensor:
        nfeats = torch.zeros_like(input.feats)
        assert input.shape[0] == 1

        # for k in range(input.shape[0]):
        #     bfeats = input.feats
        #     bfeats = bfeats.permute(1, 0).reshape(1, input.shape[1], -1)
        #     bfeats = super().forward(bfeats)
        #     bfeats = bfeats.reshape(input.shape[1], -1).permute(1, 0)
        #     nfeats = bfeats # xxxx_3333

        # assert (nfeats - bfeats).abs().max() < 0.01
        # return input.replace(nfeats, input.coords)

        bfeats = input.feats
        bfeats = bfeats.permute(1, 0).reshape(1, input.shape[1], -1)
        bfeats = super().forward(bfeats)
        bfeats = bfeats.reshape(input.shape[1], -1).permute(1, 0)
        return input.replace(bfeats)


class SparseGroupNorm32(SparseGroupNorm):
    """
    A GroupNorm layer that converts to float32 before the forward pass.
    """
    def forward(self, x: SparseTensor) -> SparseTensor:
        return super().forward(x.float()).type(x.dtype)

class SparseLayerNorm32(SparseLayerNorm):
    """
    A LayerNorm layer that converts to float32 before the forward pass.
    """
    def forward(self, x: SparseTensor) -> SparseTensor:
        return super().forward(x.float()).type(x.dtype)
