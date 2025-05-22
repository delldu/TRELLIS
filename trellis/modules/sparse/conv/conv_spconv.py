import torch
import torch.nn as nn
from .. import SparseTensor
import spconv.pytorch as spconv

import pdb

class SparseConv3d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, dilation=1, bias=True, indice_key=None):
        super().__init__()
        # in_channels = 768
        # out_channels = 192
        # kernel_size = 3 or 1
        assert stride == 1
        assert dilation == 1
        assert bias == True
        # indice_key = 'res_128'
        self.conv = spconv.SubMConv3d(in_channels, out_channels, kernel_size, dilation=dilation, bias=bias, 
            indice_key=indice_key, algo=spconv.ConvAlgo.Native)
        self.stride = (stride, stride, stride)
        
    def forward(self, x: SparseTensor) -> SparseTensor:
        new_data = self.conv(x.data)

        new_shape = [x.shape[0], self.conv.out_channels]
        out = SparseTensor(
            new_data.features, new_data.indices,
            torch.Size(new_shape),  # shape
            tuple([s * stride for s, stride in zip(x.scale, self.stride)]), # scale
            x.spatial_cache, # spatial_cache
        )
        return out
