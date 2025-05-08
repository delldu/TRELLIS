from typing import *
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from ..modules.utils import zero_module, convert_module_to_f16, convert_module_to_f32
from ..modules.transformer import AbsolutePositionEmbedder
from ..modules.norm import LayerNorm32
from ..modules import sparse as sp
from ..modules.sparse.transformer import ModulatedSparseTransformerCrossBlock
from .sparse_structure_flow import TimestepEmbedder
# from .sparse_elastic_mixin import SparseTransformerElasticMixin
import pdb

class SparseResBlock3d(nn.Module):
    def __init__(
        self,
        channels: int,
        emb_channels: int,
        out_channels: Optional[int] = None,
        downsample: bool = False,
        upsample: bool = False,
    ):
        super().__init__()
        #pdb.set_trace()

        self.channels = channels
        self.emb_channels = emb_channels
        self.out_channels = out_channels or channels
        self.downsample = downsample
        self.upsample = upsample
        
        assert not (downsample and upsample), "Cannot downsample and upsample at the same time"

        self.norm1 = LayerNorm32(channels, elementwise_affine=True, eps=1e-6)
        self.norm2 = LayerNorm32(self.out_channels, elementwise_affine=False, eps=1e-6)
        self.conv1 = sp.SparseConv3d(channels, self.out_channels, 3)
        self.conv2 = zero_module(sp.SparseConv3d(self.out_channels, self.out_channels, 3))
        self.emb_layers = nn.Sequential(
            nn.SiLU(),
            nn.Linear(emb_channels, 2 * self.out_channels, bias=True),
        )
        self.skip_connection = sp.SparseLinear(channels, self.out_channels) \
            if channels != self.out_channels else nn.Identity()
        self.updown = None
        if self.downsample:
            self.updown = sp.SparseDownsample(2)
        elif self.upsample:
            self.updown = sp.SparseUpsample(2)
        # pdb.set_trace()

    def _updown(self, x: sp.SparseTensor) -> sp.SparseTensor:
        if self.updown is not None:
            x = self.updown(x)
        return x

    def forward(self, x: sp.SparseTensor, emb: torch.Tensor) -> sp.SparseTensor:
        emb_out = self.emb_layers(emb).type(x.dtype)
        scale, shift = torch.chunk(emb_out, 2, dim=1)

        x = self._updown(x)
        h2 = x.replace(self.norm1.float()(x.feats))
        h2 = h2.replace(F.silu(h2.feats))
        h2 = self.conv1(h2)
        h2 = h2.replace(self.norm2(h2.feats)) * (1 + scale) + shift
        h2 = h2.replace(F.silu(h2.feats))
        h2 = self.conv2(h2)
        h2 = h2 + self.skip_connection(x)

        return h2
    

class SLatFlowModel(nn.Module):
    '''
    "name": "SLatFlowModel",
    "args": {
        "resolution": 64,
        "in_channels": 8,
        "out_channels": 8,
        "model_channels": 1024,
        "cond_channels": 1024,
        "num_blocks": 24,
        "num_heads": 16,
        "mlp_ratio": 4,
        "patch_size": 2,
        "num_io_res_blocks": 2,
        "io_block_channels": [128],
        "pe_mode": "ape",
        "qk_rms_norm": true,
        "use_fp16": true
    }
    '''
    def __init__(
        self,
        resolution: int,
        in_channels: int,
        model_channels: int,
        cond_channels: int,
        out_channels: int,
        num_blocks: int,
        num_heads: Optional[int] = None,
        num_head_channels: Optional[int] = 64,
        mlp_ratio: float = 4,
        patch_size: int = 2,
        num_io_res_blocks: int = 2,
        io_block_channels: List[int] = None,
        pe_mode: Literal["ape", "rope"] = "ape",
        use_fp16: bool = False,
        use_checkpoint: bool = False,
        use_skip_connection: bool = True,
        share_mod: bool = False,
        qk_rms_norm: bool = False,
        qk_rms_norm_cross: bool = False,
    ):
        super().__init__()
        # xxxx_debug
        assert resolution == 64
        assert in_channels == 8
        assert model_channels == 1024
        assert cond_channels == 1024
        assert out_channels == 8
        assert num_blocks == 24
        assert num_heads == 16
        # assert num_head_channels == 64 or ...
        assert mlp_ratio == 4
        assert patch_size == 2
        assert num_io_res_blocks == 2
        assert io_block_channels == [128]
        assert pe_mode == 'ape'
        assert use_fp16 == True
        assert use_checkpoint == False
        # assert use_skip_connection == True or ...
        # assert share_mod == False or ...
        assert qk_rms_norm == True
        # assert qk_rms_norm_cross == False or ...

        # self.resolution = resolution
        self.in_channels = in_channels # keep !!!
        # self.model_channels = model_channels
        # self.cond_channels = cond_channels
        self.out_channels = out_channels
        # self.num_blocks = num_blocks
        self.num_heads = num_heads or model_channels // num_head_channels
        # self.mlp_ratio = mlp_ratio
        # self.patch_size = patch_size
        # self.num_io_res_blocks = num_io_res_blocks
        # self.io_block_channels = io_block_channels
        self.pe_mode = pe_mode
        # self.use_fp16 = use_fp16
        # self.use_checkpoint = use_checkpoint
        self.use_skip_connection = use_skip_connection
        self.share_mod = share_mod
        # self.qk_rms_norm = qk_rms_norm
        # self.qk_rms_norm_cross = qk_rms_norm_cross
        self.dtype = torch.float16 if use_fp16 else torch.float32

        if io_block_channels is not None: # True
            assert int(np.log2(patch_size)) == np.log2(patch_size), "Patch size must be a power of 2"
            assert np.log2(patch_size) == len(io_block_channels), "Number of IO ResBlocks must match the number of stages"

        self.t_embedder = TimestepEmbedder(model_channels)
        if share_mod: # False
            self.adaLN_modulation = nn.Sequential(
                nn.SiLU(),
                nn.Linear(model_channels, 6 * model_channels, bias=True)
            )

        if pe_mode == "ape": # True
            self.pos_embedder = AbsolutePositionEmbedder(model_channels)

        self.input_layer = sp.SparseLinear(in_channels, model_channels if io_block_channels is None else io_block_channels[0])
        
        self.input_blocks = nn.ModuleList([])

        # io_block_channels == [128], [model_channels] == [1024]
        if io_block_channels is not None: # True
            for chs, next_chs in zip(io_block_channels, io_block_channels[1:] + [model_channels]):
                self.input_blocks.extend([
                    SparseResBlock3d(
                        chs,
                        model_channels,
                        out_channels=chs,
                    )
                    for _ in range(num_io_res_blocks-1) # num_io_res_blocks === 2
                ])
                self.input_blocks.append(
                    SparseResBlock3d(
                        chs,
                        model_channels,
                        out_channels=next_chs,
                        downsample=True,
                    )
                )
            
        self.blocks = nn.ModuleList([
            ModulatedSparseTransformerCrossBlock(
                model_channels,
                cond_channels,
                num_heads=self.num_heads,
                mlp_ratio=mlp_ratio,
                attn_mode='full',
                use_checkpoint=use_checkpoint,
                use_rope=(pe_mode == "rope"),
                share_mod=self.share_mod,
                qk_rms_norm=qk_rms_norm,
                qk_rms_norm_cross=qk_rms_norm_cross,
            )
            for _ in range(num_blocks)
        ])

        self.out_blocks = nn.ModuleList([])

        if io_block_channels is not None:
            for chs, prev_chs in zip(reversed(io_block_channels), [model_channels] + list(reversed(io_block_channels[1:]))):
                self.out_blocks.append(
                    SparseResBlock3d(
                        prev_chs * 2 if self.use_skip_connection else prev_chs,
                        model_channels,
                        out_channels=chs,
                        upsample=True,
                    )
                )
                self.out_blocks.extend([
                    SparseResBlock3d(
                        chs * 2 if self.use_skip_connection else chs, # self.use_skip_connection === True
                        model_channels,
                        out_channels=chs,
                    )
                    for _ in range(num_io_res_blocks-1)
                ])
            
        self.out_layer = sp.SparseLinear(model_channels if io_block_channels is None else io_block_channels[0], out_channels)

        # self.initialize_weights()
        if use_fp16:
            self.convert_to_fp16()

    @property
    def device(self) -> torch.device:
        """
        Return the device of the model.
        """
        return next(self.parameters()).device

    def convert_to_fp16(self) -> None:
        """
        Convert the torso of the model to float16.
        """
        self.input_blocks.apply(convert_module_to_f16)
        self.blocks.apply(convert_module_to_f16)
        self.out_blocks.apply(convert_module_to_f16)

    def convert_to_fp32(self) -> None:
        """
        Convert the torso of the model to float32.
        """
        self.input_blocks.apply(convert_module_to_f32)
        self.blocks.apply(convert_module_to_f32)
        self.out_blocks.apply(convert_module_to_f32)

    # def initialize_weights(self) -> None:
    #     # Initialize transformer layers:
    #     def _basic_init(module):
    #         if isinstance(module, nn.Linear):
    #             torch.nn.init.xavier_uniform_(module.weight)
    #             if module.bias is not None:
    #                 nn.init.constant_(module.bias, 0)
    #     self.apply(_basic_init)

    #     # Initialize timestep embedding MLP:
    #     nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
    #     nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

    #     # Zero-out adaLN modulation layers in DiT blocks:
    #     if self.share_mod:
    #         nn.init.constant_(self.adaLN_modulation[-1].weight, 0)
    #         nn.init.constant_(self.adaLN_modulation[-1].bias, 0)
    #     else:
    #         for block in self.blocks:
    #             nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
    #             nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

    #     # Zero-out output layers:
    #     nn.init.constant_(self.out_layer.weight, 0)
    #     nn.init.constant_(self.out_layer.bias, 0)

    def forward(self, x: sp.SparseTensor, t: torch.Tensor, cond: torch.Tensor) -> sp.SparseTensor:
        h2 = self.input_layer.float()(x).type(self.dtype)
        t_emb = self.t_embedder.float()(t)
        if self.share_mod: # False
            t_emb = self.adaLN_modulation(t_emb)

        # pdb.set_trace()
        t_emb = t_emb.type(self.dtype)
        cond = cond.type(self.dtype)

        skips = []
        # pack with input blocks
        for block in self.input_blocks:
            h2 = block(h2, t_emb)
            skips.append(h2.feats)

        # pdb.set_trace()
        if self.pe_mode == "ape": # True
            h2 = h2 + self.pos_embedder(h2.coords[:, 1:]).type(self.dtype)
        for block in self.blocks:
            h2 = block(h2, t_emb, cond)

        # pdb.set_trace()
        # unpack with output blocks
        for block, skip in zip(self.out_blocks, reversed(skips)):
            if self.use_skip_connection: # True
                h2 = block(h2.replace(torch.cat([h2.feats, skip], dim=1)), t_emb)
            else:
                h2 = block(h2, t_emb)

        # pdb.set_trace()

        h2 = h2.replace(F.layer_norm(h2.feats, h2.feats.shape[-1:]))
        h2 = self.out_layer.float()(h2.type(x.dtype))

        # pdb.set_trace()

        return h2
    

# class ElasticSLatFlowModel(SparseTransformerElasticMixin, SLatFlowModel):
#     """
#     SLat Flow Model with elastic memory management.
#     Used for training with low VRAM.
#     """
#     pass
