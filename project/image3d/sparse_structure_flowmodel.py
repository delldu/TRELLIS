import os

from typing import *
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from abs_position_embed import AbsolutePositionEmbedder
from sparse_structure_decoder import LayerNorm32
import todos
import pdb


def patchify(x: torch.Tensor, patch_size: int):
    """
    Patchify a tensor.

    Args:
        x (torch.Tensor): (N, C, *spatial) tensor
        patch_size (int): Patch size
    """
    DIM = x.dim() - 2
    for d in range(2, DIM + 2):
        assert x.shape[d] % patch_size == 0, f"Dimension {d} of input tensor must be divisible by patch size, got {x.shape[d]} and {patch_size}"

    x = x.reshape(*x.shape[:2], *sum([[x.shape[d] // patch_size, patch_size] for d in range(2, DIM + 2)], []))
    x = x.permute(0, 1, *([2 * i + 3 for i in range(DIM)] + [2 * i + 2 for i in range(DIM)]))
    x = x.reshape(x.shape[0], x.shape[1] * (patch_size ** DIM), *(x.shape[-DIM:]))
    return x

def unpatchify(x: torch.Tensor, patch_size: int):
    """
    Unpatchify a tensor.

    Args:
        x (torch.Tensor): (N, C, *spatial) tensor
        patch_size (int): Patch size
    """
    DIM = x.dim() - 2
    assert x.shape[1] % (patch_size ** DIM) == 0, f"Second dimension of input tensor must be divisible by patch size to unpatchify, got {x.shape[1]} and {patch_size ** DIM}"

    x = x.reshape(x.shape[0], x.shape[1] // (patch_size ** DIM), *([patch_size] * DIM), *(x.shape[-DIM:]))
    x = x.permute(0, 1, *(sum([[2 + DIM + i, 2 + i] for i in range(DIM)], [])))
    x = x.reshape(x.shape[0], x.shape[1], *[x.shape[2 + 2 * i] * patch_size for i in range(DIM)])
    return x


class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.

        Args:
            t: a 1-D Tensor of N indices, one per batch element.
                These may be fractional.
            dim: the dimension of the output.
            max_period: controls the minimum frequency of the embeddings.

        Returns:
            an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(
            -np.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb

class FeedForwardNet(nn.Module):
    def __init__(self, channels: int, mlp_ratio: float = 4.0):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(channels, int(channels * mlp_ratio)),
            nn.GELU(approximate="tanh"),
            nn.Linear(int(channels * mlp_ratio), channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)

class MultiHeadRMSNorm(nn.Module):
    def __init__(self, dim: int, heads: int):
        super().__init__()
        self.scale = dim ** 0.5
        self.gamma = nn.Parameter(torch.ones(heads, dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (F.normalize(x.float(), dim = -1) * self.gamma * self.scale).to(x.dtype)

def scaled_dot_product_attention(*args, **kwargs):
    arg_names_dict = {
        1: ['qkv'],
        2: ['q', 'kv'],
        3: ['q', 'k', 'v']
    }
    num_all_args = len(args) + len(kwargs)
    assert num_all_args in arg_names_dict, f"Invalid number of arguments, got {num_all_args}, expected 1, 2, or 3"
    for key in arg_names_dict[num_all_args][len(args):]:
        assert key in kwargs, f"Missing argument {key}"

    if num_all_args == 1:
        qkv = args[0] if len(args) > 0 else kwargs['qkv']
        assert len(qkv.shape) == 5 and qkv.shape[2] == 3, f"Invalid shape for qkv, got {qkv.shape}, expected [N, L, 3, H, C]"
        device = qkv.device

    elif num_all_args == 2:
        q = args[0] if len(args) > 0 else kwargs['q']
        kv = args[1] if len(args) > 1 else kwargs['kv']
        assert q.shape[0] == kv.shape[0], f"Batch size mismatch, got {q.shape[0]} and {kv.shape[0]}"
        assert len(q.shape) == 4, f"Invalid shape for q, got {q.shape}, expected [N, L, H, C]"
        assert len(kv.shape) == 5, f"Invalid shape for kv, got {kv.shape}, expected [N, L, 2, H, C]"
        device = q.device

    elif num_all_args == 3:
        q = args[0] if len(args) > 0 else kwargs['q']
        k = args[1] if len(args) > 1 else kwargs['k']
        v = args[2] if len(args) > 2 else kwargs['v']
        assert q.shape[0] == k.shape[0] == v.shape[0], f"Batch size mismatch, got {q.shape[0]}, {k.shape[0]}, and {v.shape[0]}"
        assert len(q.shape) == 4, f"Invalid shape for q, got {q.shape}, expected [N, L, H, Ci]"
        assert len(k.shape) == 4, f"Invalid shape for k, got {k.shape}, expected [N, L, H, Ci]"
        assert len(v.shape) == 4, f"Invalid shape for v, got {v.shape}, expected [N, L, H, Co]"
        device = q.device    



    if num_all_args == 1:
        q, k, v = qkv.unbind(dim=2)
    elif num_all_args == 2:
        k, v = kv.unbind(dim=2)
    out = xops.memory_efficient_attention(q, k, v)        
    
    return out



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
        self.use_rope = use_rope
        self.qk_rms_norm = qk_rms_norm

        if self._type == "self":
            self.to_qkv = nn.Linear(channels, channels * 3, bias=qkv_bias)
        else:
            self.to_q = nn.Linear(channels, channels, bias=qkv_bias)
            self.to_kv = nn.Linear(self.ctx_channels, channels * 2, bias=qkv_bias)

        # assert self.qk_rms_norm == True or ...           
        if self.qk_rms_norm: # True or False
            self.q_rms_norm = MultiHeadRMSNorm(self.head_dim, num_heads)
            self.k_rms_norm = MultiHeadRMSNorm(self.head_dim, num_heads)
            
        self.to_out = nn.Linear(channels, channels)

    
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
    ):
        super().__init__()
        assert attn_mode == "full"
        assert use_rope == False
        assert qk_rms_norm == True
        assert qkv_bias == True
        
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
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(channels, 6 * channels, bias=True)
        )
        
    def _forward(self, x: torch.Tensor, mod: torch.Tensor, context: torch.Tensor):
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

# xxxx_1111
class SparseStructureFlowModel(nn.Module):
    '''
    "name": "SparseStructureFlowModel",
    "args": {
        "resolution": 16,
        "in_channels": 8,
        "out_channels": 8,
        "model_channels": 1024,
        "cond_channels": 1024,
        "num_blocks": 24,
        "num_heads": 16,
        "mlp_ratio": 4,
        "patch_size": 1,
        "pe_mode": "ape",
        "qk_rms_norm": true,
    }
    '''    
    def __init__(
        self,
        resolution = 16,
        in_channels = 8,
        model_channels = 1024,
        cond_channels = 1024,
        out_channels = 8,
        num_blocks = 24,
        num_heads = 16,
        num_head_channels = 64,
        mlp_ratio = 4,
        patch_size = 1,
        pe_mode = "ape",
        qk_rms_norm = True,
        qk_rms_norm_cross = False,
    ):
        super().__init__()
        # print(f"SparseStructureFlowModel: num_head_channels={num_head_channels}, qk_rms_norm_cross={qk_rms_norm_cross}")
        # SparseStructureFlowModel: num_head_channels=64, qk_rms_norm_cross=False

        assert resolution == 16
        assert in_channels == 8
        assert model_channels == 1024
        assert cond_channels == 1024
        assert out_channels == 8
        assert num_blocks == 24
        assert num_heads == 16
        assert num_head_channels == 64
        assert mlp_ratio == 4
        assert patch_size == 1
        assert pe_mode == 'ape'
        assert qk_rms_norm == True
        assert qk_rms_norm_cross == False

        self.resolution = resolution
        self.in_channels = in_channels
        self.patch_size = patch_size
        self.dtype = torch.float16

        self.t_embedder = TimestepEmbedder(model_channels)

        if pe_mode == "ape": # True
            pos_embedder = AbsolutePositionEmbedder(model_channels, 3)
            coords = torch.meshgrid(*[torch.arange(res) for res in [resolution // patch_size] * 3], indexing='ij')
            coords = torch.stack(coords, dim=-1).reshape(-1, 3)
            pos_emb = pos_embedder(coords)
            self.register_buffer("pos_emb", pos_emb)

        self.input_layer = nn.Linear(in_channels * patch_size**3, model_channels)
            
        self.blocks = nn.ModuleList([
            ModulatedTransformerCrossBlock(
                model_channels,
                cond_channels,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                attn_mode='full',
                use_rope=(pe_mode == "rope"),
                qk_rms_norm=qk_rms_norm,
                qk_rms_norm_cross=qk_rms_norm_cross,
            )
            for _ in range(num_blocks)
        ])

        self.out_layer = nn.Linear(model_channels, out_channels * patch_size**3)

    def load_weights(self, model_path="models/image3d_dinov2.pth"):
        cdir = os.path.dirname(__file__)
        checkpoint = model_path if cdir == "" else cdir + "/" + model_path
        print(f"Loading {checkpoint} ...")
        self.load_state_dict(torch.load(checkpoint), strict=True)


    def forward(self, x: torch.Tensor, t: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        # assert [*x.shape] == [x.shape[0], self.in_channels, *[self.resolution] * 3], \
        #         f"Input shape mismatch, got {x.shape}, expected {[x.shape[0], self.in_channels, *[self.resolution] * 3]}"
        # tensor [x] size: [1, 8, 16, 16, 16], min: -4.184124, max: 3.802687, mean: 0.000481
        # tensor [t] size: [1], min: 1000.0, max: 1000.0, mean: 1000.0
        # tensor [cond] size: [1, 1374, 1024], min: -25.644331, max: 15.487422, mean: 0.0

        h2 = patchify(x, self.patch_size)
        # tensor [h2] size: [1, 8, 16, 16, 16], min: -4.184124, max: 3.802687, mean: 0.000481
        h2 = h2.view(*h2.shape[:2], -1).permute(0, 2, 1).contiguous()
        # tensor [h2] size: [1, 4096, 8], min: -4.184124, max: 3.802687, mean: 0.000481

        h2 = self.input_layer(h2.type(self.dtype))

        # tensor [h2] size: [1, 4096, 1024], min: -3.353516, max: 3.339844, mean: -0.014493
        # tensor [self.pos_emb[None]] size: [1, 4096, 1024], min: -1.0, max: 1.0, mean: 0.454115
        h2 = h2 + self.pos_emb[None] # cuda, half float ...
        # tensor [h2] size: [1, 4096, 1024], min: -3.283203, max: 4.097656, mean: 0.439621

        t_emb = self.t_embedder.float()(t)
        t_emb = t_emb.type(self.dtype)

        h2 = h2.type(self.dtype)
        cond = cond.type(self.dtype)
        for block in self.blocks:
            h2 = block.float()(h2.float(), t_emb.float(), cond.float()).type(self.dtype)

        h2 = h2.type(x.dtype)
        # tensor [h2] size: [1, 4096, 1024], min: -4260.0, max: 6168.0, mean: 6.813056

        h2 = F.layer_norm(h2, h2.shape[-1:]) # h2.shape[-1:] -- 1024
        h2 = self.out_layer(h2.type(self.dtype))

        # tensor [h2] size: [1, 4096, 8], min: -4.324219, max: 3.894531, mean: -0.007873
        h2 = h2.permute(0, 2, 1).view(h2.shape[0], h2.shape[2], *[self.resolution // self.patch_size] * 3) # self.resolution == 16
        # tensor [h2] size: [1, 8, 16, 16, 16], min: -4.324219, max: 3.894531, mean: -0.007873

        h2 = unpatchify(h2, self.patch_size).contiguous()
        # tensor [h2] size: [1, 8, 16, 16, 16], min: -4.324219, max: 3.896484, mean: -0.007871

        return h2


if __name__ == "__main__":
    model = SparseStructureFlowModel()
    model.eval()
    
    print(model)