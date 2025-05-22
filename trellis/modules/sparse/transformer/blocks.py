from typing import *
import torch
import torch.nn as nn
from ..basic import SparseTensor
from ..linear import SparseLinear
from ..nonlinearity import SparseGELU
from ..attention import SparseMultiHeadAttention
from ...norm import LayerNorm32
import pdb

class SparseFeedForwardNet(nn.Module):
    def __init__(self, channels: int, mlp_ratio: float = 4.0):
        super().__init__()
        self.mlp = nn.Sequential(
            SparseLinear(channels, int(channels * mlp_ratio)),
            SparseGELU(approximate="tanh"),
            SparseLinear(int(channels * mlp_ratio), channels),
        )
        # channels = 1024
        # mlp_ratio = 4

    def forward(self, x: SparseTensor) -> SparseTensor:
        return self.mlp(x)


class SparseTransformerBlock(nn.Module):
    """
    Sparse Transformer block (MSA + FFN).
    """
    def __init__(
        self,
        channels: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        attn_mode: Literal["full", "shift_window", "shift_sequence", "shift_order", "swin"] = "full",
        window_size: Optional[int] = None,
        shift_sequence: Optional[int] = None,
        shift_window: Optional[Tuple[int, int, int]] = None,
        serialize_mode = None,
        use_rope: bool = False,
        qk_rms_norm: bool = False,
        qkv_bias: bool = True,
        ln_affine: bool = False,
    ):
        super().__init__()
        assert shift_sequence == None
        assert serialize_mode == None

        self.norm1 = LayerNorm32(channels, elementwise_affine=ln_affine, eps=1e-6)
        self.norm2 = LayerNorm32(channels, elementwise_affine=ln_affine, eps=1e-6)
        self.attn = SparseMultiHeadAttention(
            channels,
            num_heads=num_heads,
            attn_mode=attn_mode,
            window_size=window_size,
            shift_sequence=shift_sequence,
            shift_window=shift_window,
            serialize_mode=serialize_mode,
            qkv_bias=qkv_bias,
            use_rope=use_rope,
            qk_rms_norm=qk_rms_norm,
        )
        self.mlp = SparseFeedForwardNet(
            channels,
            mlp_ratio=mlp_ratio,
        )

    def forward(self, x: SparseTensor) -> SparseTensor:
        h = x.replace(self.norm1.float()(x.feats))
        h = self.attn(h)
        x = x + h
        h = x.replace(self.norm2.float()(x.feats))
        h = self.mlp(h)
        x = x + h
        return x
