from typing import *
import torch
import torch.nn as nn
import torch.nn.functional as F
from .full_attn import scaled_dot_product_attention
import todos
import pdb

class MultiHeadRMSNorm(nn.Module):
    def __init__(self, dim: int, heads: int):
        super().__init__()
        self.scale = dim ** 0.5
        self.gamma = nn.Parameter(torch.ones(heads, dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (F.normalize(x.float(), dim = -1) * self.gamma * self.scale).to(x.dtype)

class MultiHeadAttention(nn.Module):
    def __init__(
        self,
        channels: int,
        num_heads: int,
        ctx_channels: Optional[int]=None,
        type: Literal["self", "cross"] = "self",
        attn_mode: Literal["full", "windowed"] = "full",
        window_size: Optional[int] = None,
        shift_window: Optional[Tuple[int, int, int]] = None,
        qkv_bias: bool = True,
        use_rope: bool = False,
        qk_rms_norm: bool = False,
    ):
        super().__init__()
        # print(f"== MultiHeadAttention: type={type}, use_rope={use_rope}, attn_mode={attn_mode}, qk_rms_norm={qk_rms_norm}")
        # == MultiHeadAttention: type=self, use_rope=False, attn_mode=full, qk_rms_norm=True
        # == MultiHeadAttention: type=cross, use_rope=False, attn_mode=full, qk_rms_norm=False

        assert channels == 1024
        assert num_heads == 16
        # assert ctx_channels == None or ...
        # assert type = 'self' or 'cross''
        assert attn_mode == 'full'
        assert window_size == None
        assert shift_window == None
        # assert qkv_bias == True or ...
        assert use_rope == False
        # assert qk_rms_norm == True or ...

        assert channels % num_heads == 0
        assert type in ["self", "cross"], f"Invalid attention type: {type}"
        assert attn_mode in ["full", "windowed"], f"Invalid attention mode: {attn_mode}"
        assert type == "self" or attn_mode == "full", "Cross-attention only supports full attention"
        
        if attn_mode == "windowed":
            raise NotImplementedError("Windowed attention is not yet implemented")
        
        # self.channels = channels
        self.head_dim = channels // num_heads
        assert self.head_dim == 64

        self.ctx_channels = ctx_channels if ctx_channels is not None else channels
        assert self.ctx_channels == channels # xxxx_3333

        self.num_heads = num_heads
        self._type = type
        self.attn_mode = attn_mode
        # self.window_size = window_size
        # self.shift_window = shift_window
        # self.use_rope = use_rope
        self.qk_rms_norm = qk_rms_norm

        if self._type == "self":
            self.to_qkv = nn.Linear(channels, channels * 3, bias=qkv_bias)
        else:
            self.to_q = nn.Linear(channels, channels, bias=qkv_bias)
            self.to_kv = nn.Linear(self.ctx_channels, channels * 2, bias=qkv_bias)

        # assert self.qk_rms_norm == True or ...           
        if self.qk_rms_norm: # True
            self.q_rms_norm = MultiHeadRMSNorm(self.head_dim, num_heads)
            self.k_rms_norm = MultiHeadRMSNorm(self.head_dim, num_heads)
            
        self.to_out = nn.Linear(channels, channels)

        # if use_rope:
        #     self.rope = RotaryPositionEmbedder(channels)
    
    def forward(self, x: torch.Tensor, context: Optional[torch.Tensor] = None, indices: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, L, C = x.shape
        # assert self.qk_rms_norm == True or ...

        # tensor [x] size: [1, 4096, 1024], min: -24.621138, max: 10.970028, mean: 0.006027
        # tensor [context] size: [1, 1374, 1024], min: -25.640625, max: 15.484375, mean: 0.0
        # [context] type: <class 'NoneType'>
        # [indices] type: <class 'NoneType'>
        assert indices == None

        if self._type == "self":
            # == MultiHeadAttention: type=self, use_rope=False, attn_mode=full, qk_rms_norm=True
            qkv = self.to_qkv(x)
            qkv = qkv.reshape(B, L, 3, self.num_heads, -1)
            if self.attn_mode == "full":
                q, k, v = qkv.unbind(dim=2)
                q = self.q_rms_norm(q)
                k = self.k_rms_norm(k)
                h = scaled_dot_product_attention(q, k, v)
            elif self.attn_mode == "windowed":
                raise NotImplementedError("Windowed attention is not yet implemented")
        else:
            # == MultiHeadAttention: type=cross, use_rope=False, attn_mode=full, qk_rms_norm=False
            # tensor [context] size: [1, 1374, 1024], min: -25.640625, max: 15.484375, mean: 0.0
            Lkv = context.shape[1] # 1374
            q = self.to_q(x)
            kv = self.to_kv(context)
            q = q.reshape(B, L, self.num_heads, -1)
            kv = kv.reshape(B, Lkv, 2, self.num_heads, -1)
            h = scaled_dot_product_attention(q, kv)
        h = h.reshape(B, L, -1)
        h = self.to_out(h)

        # tensor [h] size: [1, 4096, 1024], min: -6.635851, max: 4.444528, mean: 0.003872
        return h
