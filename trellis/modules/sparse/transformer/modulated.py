from typing import *
import torch
import torch.nn as nn
from ..basic import SparseTensor
from ..attention import SparseMultiHeadAttention #, SerializeMode
from ...norm import LayerNorm32
from .blocks import SparseFeedForwardNet
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
        assert channels == 1024
        assert ctx_channels == 1024
        assert num_heads == 16
        assert mlp_ratio == 4
        assert attn_mode == 'full'
        assert window_size == None
        assert shift_sequence == None
        assert shift_window == None
        assert serialize_mode == None
        assert use_rope == False
        assert qk_rms_norm == True
        assert qk_rms_norm_cross == False
        assert qkv_bias == True
        assert share_mod == False

        self.share_mod = share_mod
        self.norm1 = LayerNorm32(channels, elementwise_affine=False, eps=1e-6)
        self.norm2 = LayerNorm32(channels, elementwise_affine=True, eps=1e-6)
        self.norm3 = LayerNorm32(channels, elementwise_affine=False, eps=1e-6)
        self.self_attn = SparseMultiHeadAttention(
            channels, # 1024
            num_heads=num_heads, # 16
            type="self",
            attn_mode=attn_mode, # "full"
            window_size=window_size,
            shift_sequence=shift_sequence, # None
            shift_window=shift_window, # None
            serialize_mode=serialize_mode, # None
            qkv_bias=qkv_bias, # True
            use_rope=use_rope, # False
            qk_rms_norm=qk_rms_norm, # True
        )
        self.cross_attn = SparseMultiHeadAttention(
            channels, # 1024
            ctx_channels=ctx_channels, # 1024
            num_heads=num_heads, # 16
            type="cross",
            attn_mode="full",
            qkv_bias=qkv_bias, # True
            qk_rms_norm=qk_rms_norm_cross, # False
        )
        self.mlp = SparseFeedForwardNet(
            channels,
            mlp_ratio=mlp_ratio,
        )
        if not share_mod: # True
            self.adaLN_modulation = nn.Sequential(
                nn.SiLU(),
                nn.Linear(channels, 6 * channels, bias=True)
            )

    def forward(self, x: SparseTensor, mod: torch.Tensor, context: torch.Tensor) -> SparseTensor:
        if self.share_mod: # False
            pdb.set_trace()
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = mod.chunk(6, dim=1)
        else:
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(mod).chunk(6, dim=1)

        h = x.replace(self.norm1.float()(x.feats))
        h = h * (1 + scale_msa) + shift_msa
        h = self.self_attn(h)
        h = h * gate_msa
        x = x + h
        h = x.replace(self.norm2.float()(x.feats), x.coords)
        h = self.cross_attn(h, context)
        x = x + h
        h = x.replace(self.norm3.float()(x.feats), x.coords)
        h = h * (1 + scale_mlp) + shift_mlp
        h = self.mlp(h)
        h = h * gate_mlp
        x = x + h

        return x

