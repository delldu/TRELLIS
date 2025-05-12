import os

from typing import *
import torch
import torch.nn as nn
import torch.nn.functional as F
from abs_position_embed import AbsolutePositionEmbedder
from sparse_structure_decoder import LayerNorm32
from sparse_tensor import (
    SparseTensor, 
    SparseConv3d, 
    SparseDownsample, 
    SparseUpsample, 
    SparseLinear, 
    SparseMultiHeadAttention,
    SparseFeedForwardNet,
)
from sparse_structure_flowmodel import TimestepEmbedder
import pdb

class ModulatedSparseTransformerCrossBlock(nn.Module):
    """
    Sparse Transformer cross-attention block (MSA + MCA + FFN) with adaptive layer norm conditioning.
    """
    def __init__(
        self,
        channels: int,
        ctx_channels: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        attn_mode: Literal["full", "shift_window", "shift_sequence", "shift_order", "swin"] = "full",
        window_size: Optional[int] = None,
        shift_sequence: Optional[int] = None,
        shift_window: Optional[Tuple[int, int, int]] = None,
        serialize_mode = None,
        use_rope: bool = False,
        qk_rms_norm: bool = False,
        qk_rms_norm_cross: bool = False,
        qkv_bias: bool = True,
        share_mod: bool = False,

    ):
        super().__init__()
        self.share_mod = share_mod
        self.norm1 = LayerNorm32(channels, elementwise_affine=False, eps=1e-6)
        self.norm2 = LayerNorm32(channels, elementwise_affine=True, eps=1e-6)
        self.norm3 = LayerNorm32(channels, elementwise_affine=False, eps=1e-6)
        self.self_attn = SparseMultiHeadAttention(
            channels,
            num_heads=num_heads,
            type="self",
            attn_mode=attn_mode,
            window_size=window_size,
            shift_sequence=shift_sequence,
            shift_window=shift_window,
            serialize_mode=serialize_mode,
            qkv_bias=qkv_bias,
            use_rope=use_rope,
            qk_rms_norm=qk_rms_norm,
        )
        self.cross_attn = SparseMultiHeadAttention(
            channels,
            ctx_channels=ctx_channels,
            num_heads=num_heads,
            type="cross",
            attn_mode="full",
            qkv_bias=qkv_bias,
            qk_rms_norm=qk_rms_norm_cross,
        )
        self.mlp = SparseFeedForwardNet(
            channels,
            mlp_ratio=mlp_ratio,
        )
        if not share_mod:
            self.adaLN_modulation = nn.Sequential(
                nn.SiLU(),
                nn.Linear(channels, 6 * channels, bias=True)
            )

    def forward(self, x: SparseTensor, mod: torch.Tensor, context: torch.Tensor) -> SparseTensor:
        if self.share_mod:
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = mod.chunk(6, dim=1)
        else:
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(mod).chunk(6, dim=1)

        h = x.replace(self.norm1.float()(x.feats))
        h = h * (1 + scale_msa) + shift_msa
        h = self.self_attn(h)
        h = h * gate_msa
        x = x + h
        h = x.replace(self.norm2.float()(x.feats))
        h = self.cross_attn(h, context)
        x = x + h
        h = x.replace(self.norm3.float()(x.feats))
        h = h * (1 + scale_mlp) + shift_mlp
        h = self.mlp(h)
        h = h * gate_mlp
        x = x + h

        return x


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

        self.channels = channels
        self.emb_channels = emb_channels
        self.out_channels = out_channels or channels
        self.downsample = downsample
        self.upsample = upsample
        
        assert not (downsample and upsample), "Cannot downsample and upsample at the same time"

        self.norm1 = LayerNorm32(channels, elementwise_affine=True, eps=1e-6)
        self.norm2 = LayerNorm32(self.out_channels, elementwise_affine=False, eps=1e-6)
        self.conv1 = SparseConv3d(channels, self.out_channels, 3)
        self.conv2 = SparseConv3d(self.out_channels, self.out_channels, 3)
        self.emb_layers = nn.Sequential(
            nn.SiLU(),
            nn.Linear(emb_channels, 2 * self.out_channels, bias=True),
        )
        self.skip_connection = SparseLinear(channels, self.out_channels) \
            if channels != self.out_channels else nn.Identity()
        self.updown = None
        if self.downsample:
            self.updown = SparseDownsample(2)
        elif self.upsample:
            self.updown = SparseUpsample(2)
        # pdb.set_trace()

    def _updown(self, x: SparseTensor) -> SparseTensor:
        if self.updown is not None:
            x = self.updown(x)
        return x

    def forward(self, x: SparseTensor, emb: torch.Tensor) -> SparseTensor:
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
    
# xxxx_1111
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
    }
    '''
    def __init__(
        self,
        resolution = 64,
        in_channels = 8,
        model_channels = 1024,
        cond_channels = 1024,
        out_channels = 8,
        num_blocks = 24,
        num_heads = 16,
        num_head_channels = 64,
        mlp_ratio = 4,
        patch_size = 2,
        num_io_res_blocks = 2,
        io_block_channels = [128],
        pe_mode = "ape",
        use_skip_connection = True,
        share_mod = False,
        qk_rms_norm = True,
        qk_rms_norm_cross = False,
    ):
        super().__init__()
        # print(f"SLatFlowModel: num_head_channels={num_head_channels}, use_skip_connection={use_skip_connection}, share_mod={share_mod}, qk_rms_norm_cross={qk_rms_norm_cross}")
        # SLatFlowModel: num_head_channels=64, use_skip_connection=True, share_mod=False, qk_rms_norm_cross=False

        assert resolution == 64
        assert in_channels == 8
        assert model_channels == 1024
        assert cond_channels == 1024
        assert out_channels == 8
        assert num_blocks == 24
        assert num_heads == 16
        assert num_head_channels == 64
        assert mlp_ratio == 4
        assert patch_size == 2
        assert num_io_res_blocks == 2
        assert io_block_channels == [128]
        assert pe_mode == 'ape'
        assert use_skip_connection == True
        assert share_mod == False
        assert qk_rms_norm == True
        assert qk_rms_norm_cross == False

        self.in_channels = in_channels # keep it !!!
        self.out_channels = out_channels
        self.num_heads = num_heads or model_channels // num_head_channels
        self.pe_mode = pe_mode
        self.use_skip_connection = use_skip_connection
        self.share_mod = share_mod
        self.dtype = torch.float16

        self.t_embedder = TimestepEmbedder(model_channels)
        if share_mod: # False
            self.adaLN_modulation = nn.Sequential(
                nn.SiLU(),
                nn.Linear(model_channels, 6 * model_channels, bias=True)
            )

        if pe_mode == "ape": # True
            self.pos_embedder = AbsolutePositionEmbedder(model_channels)

        self.input_layer = SparseLinear(in_channels, io_block_channels[0])
        
        self.input_blocks = nn.ModuleList([])

        # io_block_channels == [128], [model_channels] == [1024]
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
                use_rope=(pe_mode == "rope"),
                share_mod=self.share_mod,
                qk_rms_norm=qk_rms_norm,
                qk_rms_norm_cross=qk_rms_norm_cross,
            )
            for _ in range(num_blocks)
        ])

        self.out_blocks = nn.ModuleList([])

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
            
        self.out_layer = SparseLinear(io_block_channels[0], out_channels)

    def load_weights(self, model_path="models/image3d_dinov2.pth"):
        cdir = os.path.dirname(__file__)
        checkpoint = model_path if cdir == "" else cdir + "/" + model_path
        print(f"Loading {checkpoint} ...")
        self.load_state_dict(torch.load(checkpoint), strict=True)
        
    # xxxx_debug
    def forward(self, x: SparseTensor, t: torch.Tensor, cond: torch.Tensor) -> SparseTensor:
        # tensor [x data.coords] size: [14955, 4], min: 0.0, max: 63.0, mean: 23.262018
        # tensor [x data.features] size: [14955, 8], min: -4.452163, max: 4.218491, mean: -0.000569

        h2 = self.input_layer.float()(x).type(self.dtype)
        t_emb = self.t_embedder.float()(t)
        if self.share_mod: # False
            t_emb = self.adaLN_modulation(t_emb)

        t_emb = t_emb.type(self.dtype)
        cond = cond.type(self.dtype)

        skips = []
        # pack with input blocks
        for block in self.input_blocks:
            h2 = block(h2, t_emb)
            skips.append(h2.feats)

        if self.pe_mode == "ape": # True
            h2 = h2 + self.pos_embedder(h2.coords[:, 1:]).type(self.dtype)
        for block in self.blocks:
            h2 = block(h2, t_emb, cond)

        # unpack with output blocks
        for block, skip in zip(self.out_blocks, reversed(skips)):
            if self.use_skip_connection: # True
                h2 = block(h2.replace(torch.cat([h2.feats, skip], dim=1)), t_emb)
            else:
                h2 = block(h2, t_emb)

        h2 = h2.replace(F.layer_norm(h2.feats, h2.feats.shape[-1:]))
        h2 = self.out_layer.float()(h2.type(x.dtype))

        # tensor [h2 data.coords] size: [14955, 4], min: 0.0, max: 63.0, mean: 23.262018
        # tensor [h2 data.features] size: [14955, 8], min: -6.325896, max: 5.703651, mean: -0.000702
        
        return h2


if __name__ == "__main__":
    model = SLatFlowModel()
    model.eval()
    
    print(model)