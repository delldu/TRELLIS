from typing import *
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from ...modules.utils import zero_module, convert_module_to_f16, convert_module_to_f32
from ...modules import sparse as sp
from .base import SparseTransformerBase
from ...representations.mesh import SparseFeatures2Mesh
from typing import List, Optional 

import pdb

class SparseSubdivideBlock3d(nn.Module):
    """
    A 3D subdivide block that can subdivide the sparse tensor.

    Args:
        channels: channels in the inputs and outputs.
        out_channels: if specified, the number of output channels.
        num_groups: the number of groups for the group norm.
    """
    def __init__(
        self,
        channels: int,
        resolution: int,
        out_channels: Optional[int] = None,
        num_groups: int = 32
    ):
        super().__init__()
        # channels = 768
        # resolution = 64
        # out_channels = 192
        # num_groups = 32

        # self.channels = channels
        # self.resolution = resolution
        self.out_resolution = resolution * 2
        self.out_channels = out_channels or channels

        # sp.norm.SparseGroupNorm32(num_groups, channels)
        self.act_layers = nn.Sequential(
            sp.SparseGroupNorm32(num_groups, channels), # (32, 768)
            sp.SparseSiLU()
        )
        
        self.sub = sp.SparseSubdivide()
        
        self.out_layers = nn.Sequential(
            sp.SparseConv3d(channels, self.out_channels, 3, indice_key=f"res_{self.out_resolution}"),
            sp.SparseGroupNorm32(num_groups, self.out_channels),
            sp.SparseSiLU(),
            zero_module(sp.SparseConv3d(self.out_channels, self.out_channels, 3, indice_key=f"res_{self.out_resolution}")),
        )
        
        if self.out_channels == channels:
            self.skip_connection = nn.Identity()
        else:
            self.skip_connection = sp.SparseConv3d(channels, self.out_channels, 1, indice_key=f"res_{self.out_resolution}")
        
    def forward(self, x: sp.SparseTensor) -> sp.SparseTensor:
        """
        Apply the block to a Tensor, conditioned on a timestep embedding.

        Args:
            x: an [N x C x ...] Tensor of features.
        Returns:
            an [N x C x ...] Tensor of outputs.
        """

        h2 = self.act_layers.float()(x) # SparseGroupNorm32
        h2 = self.sub(h2)
        x = self.sub(x)
        h2 = self.out_layers.float()(h2.float())
        h2 = h2 + self.skip_connection(x)

        return h2


class SLatMeshDecoder(SparseTransformerBase):
    def __init__(
        self,
        resolution: int,
        model_channels: int,
        latent_channels: int,
        num_blocks: int,
        num_heads: Optional[int] = None,
        num_head_channels: Optional[int] = 64,
        mlp_ratio: float = 4,
        attn_mode: Literal["full", "shift_window", "shift_sequence", "shift_order", "swin"] = "swin",
        window_size: int = 8,
        pe_mode: Literal["ape", "rope"] = "ape",
        use_fp16: bool = False,
        qk_rms_norm: bool = False,
        representation_config: dict = None,
    ):
        super().__init__(
            in_channels=latent_channels,
            model_channels=model_channels,
            num_blocks=num_blocks,
            num_heads=num_heads,
            num_head_channels=num_head_channels,
            mlp_ratio=mlp_ratio,
            attn_mode=attn_mode,
            window_size=window_size,
            pe_mode=pe_mode,
            use_fp16=use_fp16,
            qk_rms_norm=qk_rms_norm,
        )
        # assert resolution == 64
        # assert model_channels == 768
        # assert latent_channels == 8
        # assert num_blocks == 12
        # assert num_heads == 12
        assert num_head_channels == 64
        assert mlp_ratio == 4
        assert attn_mode == 'swin'
        # assert window_size == 8
        # assert pe_mode == 'ape'
        assert use_fp16 == True
        # assert qk_rms_norm == False
        # representation_config = {'use_color': True}
        self.rep_config = representation_config

        self.mesh_extractor = SparseFeatures2Mesh(res=resolution*4, use_color=self.rep_config.get('use_color', False))
        self.out_channels = self.mesh_extractor.feats_channels

        self.upsample = nn.ModuleList([
            SparseSubdivideBlock3d(
                channels=model_channels, # 768
                resolution=resolution, # 64
                out_channels=model_channels // 4 # 192
            ),
            SparseSubdivideBlock3d(
                channels=model_channels // 4,
                resolution=resolution * 2,
                out_channels=model_channels // 8
            )
        ])
        self.out_layer = sp.SparseLinear(model_channels // 8, self.out_channels)

        if use_fp16:
            self.convert_to_fp16()

    def convert_to_fp16(self) -> None:
        """
        Convert the torsor of the model to float16.
        """
        super().convert_to_fp16()
        self.upsample.apply(convert_module_to_f16)

    def convert_to_fp32(self) -> None:
        """
        Convert the torsor of the model to float32.
        """
        super().convert_to_fp32()
        self.upsample.apply(convert_module_to_f32)  
    
    def to_representation(self, x: sp.SparseTensor):
        """
        Convert a batch of network outputs to 3D representations.

        Args:
            x: The [N x * x C] sparse tensor output by the network.

        Returns:
            list of representations
        """
        ret = []
        for i in range(x.shape[0]):
            mesh = self.mesh_extractor(x[i], training=self.training)
            ret.append(mesh)
        return ret

    def forward(self, x: sp.SparseTensor):
        h2 = super().forward(x)

        for block in self.upsample: # SparseSubdivideBlock3d
            h2 = block.float()(h2.float())

        h2 = h2.type(x.dtype)

        h2 = self.out_layer.float()(h2)

        return self.to_representation(h2)
    
