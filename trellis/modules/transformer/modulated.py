from typing import *
import torch
import torch.nn as nn
from ..attention import MultiHeadAttention
from ..norm import LayerNorm32
from .blocks import FeedForwardNet
import pdb

# class ModulatedTransformerBlock(nn.Module):
#     """
#     Transformer block (MSA + FFN) with adaptive layer norm conditioning.
#     """
#     def __init__(
#         self,
#         channels: int,
#         num_heads: int,
#         mlp_ratio: float = 4.0,
#         attn_mode: Literal["full", "windowed"] = "full",
#         window_size: Optional[int] = None,
#         shift_window: Optional[Tuple[int, int, int]] = None,
#         use_rope: bool = False,
#         qk_rms_norm: bool = False,
#         qkv_bias: bool = True,
#         share_mod: bool = False,
#     ):
#         super().__init__()
#         print(f"== ModulatedTransformerBlock: attn_mode={attn_mode}, use_rope={use_rope}, qk_rms_norm={qk_rms_norm}, share_mod={share_mod}")

#         self.share_mod = share_mod
#         self.norm1 = LayerNorm32(channels, elementwise_affine=False, eps=1e-6)
#         self.norm2 = LayerNorm32(channels, elementwise_affine=False, eps=1e-6)
#         self.attn = MultiHeadAttention(
#             channels,
#             num_heads=num_heads,
#             attn_mode=attn_mode,
#             window_size=window_size,
#             shift_window=shift_window,
#             qkv_bias=qkv_bias,
#             use_rope=use_rope,
#             qk_rms_norm=qk_rms_norm,
#         )
#         self.mlp = FeedForwardNet(
#             channels,
#             mlp_ratio=mlp_ratio,
#         )
#         if not share_mod:
#             self.adaLN_modulation = nn.Sequential(
#                 nn.SiLU(),
#                 nn.Linear(channels, 6 * channels, bias=True)
#             )
#         # xxxx_debug pdb.set_trace()

#     def _forward(self, x: torch.Tensor, mod: torch.Tensor) -> torch.Tensor:
#         if self.share_mod:
#             shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = mod.chunk(6, dim=1)
#         else:
#             shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(mod).chunk(6, dim=1)
#         h = self.norm1(x)
#         h = h * (1 + scale_msa.unsqueeze(1)) + shift_msa.unsqueeze(1)
#         h = self.attn(h)
#         h = h * gate_msa.unsqueeze(1)
#         x = x + h
#         h = self.norm2(x)
#         h = h * (1 + scale_mlp.unsqueeze(1)) + shift_mlp.unsqueeze(1)
#         h = self.mlp(h)
#         h = h * gate_mlp.unsqueeze(1)
#         x = x + h
#         return x

#     def forward(self, x: torch.Tensor, mod: torch.Tensor) -> torch.Tensor:
#         return self._forward(x, mod)

class ModulatedTransformerCrossBlock(nn.Module):
    """
    Transformer cross-attention block (MSA + MCA + FFN) with adaptive layer norm conditioning.
    """
    def __init__(
        self,
        channels: int,
        ctx_channels: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        attn_mode: Literal["full", "windowed"] = "full",
        window_size: Optional[int] = None,
        shift_window: Optional[Tuple[int, int, int]] = None,
        use_rope: bool = False,
        qk_rms_norm: bool = False,
        qk_rms_norm_cross: bool = False,
        qkv_bias: bool = True,
        share_mod: bool = False,
    ):
        super().__init__()
        # print(f"== ModulatedTransformerCrossBlock: attn_mode={attn_mode}, use_rope={use_rope}, qk_rms_norm={qk_rms_norm}, qkv_bias={qkv_bias}, share_mod={share_mod}")
        # == ModulatedTransformerCrossBlock: attn_mode=full, use_rope=False, qk_rms_norm=True, qkv_bias=True, share_mod=False

        assert attn_mode == "full"
        assert use_rope == False
        assert qk_rms_norm == True
        assert qkv_bias == True
        assert share_mod == False
        
        self.share_mod = share_mod
        self.norm1 = LayerNorm32(channels, elementwise_affine=False, eps=1e-6)
        self.norm2 = LayerNorm32(channels, elementwise_affine=True, eps=1e-6)
        self.norm3 = LayerNorm32(channels, elementwise_affine=False, eps=1e-6)
        self.self_attn = MultiHeadAttention(
            channels,
            num_heads=num_heads,
            type="self",
            attn_mode=attn_mode,
            window_size=window_size,
            shift_window=shift_window,
            qkv_bias=qkv_bias,
            use_rope=use_rope,
            qk_rms_norm=qk_rms_norm,
        )
        self.cross_attn = MultiHeadAttention(
            channels,
            ctx_channels=ctx_channels,
            num_heads=num_heads,
            type="cross",
            attn_mode="full",
            qkv_bias=qkv_bias,
            qk_rms_norm=qk_rms_norm_cross,
        )
        self.mlp = FeedForwardNet(
            channels,
            mlp_ratio=mlp_ratio,
        )
        if not share_mod:
            self.adaLN_modulation = nn.Sequential(
                nn.SiLU(),
                nn.Linear(channels, 6 * channels, bias=True)
            )
        # xxxx_debug pdb.set_trace() ???
        
    def _forward(self, x: torch.Tensor, mod: torch.Tensor, context: torch.Tensor):
        if self.share_mod: # False
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = mod.chunk(6, dim=1)
        else:
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(mod).chunk(6, dim=1)
        h = self.norm1(x)
        h = h * (1 + scale_msa.unsqueeze(1)) + shift_msa.unsqueeze(1)
        h = self.self_attn(h)
        h = h * gate_msa.unsqueeze(1)
        x = x + h
        h = self.norm2(x)
        h = self.cross_attn(h, context)
        x = x + h
        h = self.norm3(x)
        h = h * (1 + scale_mlp.unsqueeze(1)) + shift_mlp.unsqueeze(1)
        h = self.mlp(h)
        h = h * gate_mlp.unsqueeze(1)
        x = x + h
        return x

    def forward(self, x: torch.Tensor, mod: torch.Tensor, context: torch.Tensor):
        return self._forward(x, mod, context)
        