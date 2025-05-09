from typing import *
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from ..modules.utils import convert_module_to_f16, convert_module_to_f32
from ..modules.transformer import AbsolutePositionEmbedder, ModulatedTransformerCrossBlock
from ..modules.spatial import patchify, unpatchify
import todos
import pdb

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
        pe_mode: Literal["ape", "rope"] = "ape",
        use_fp16: bool = False,
        share_mod: bool = False,
        qk_rms_norm: bool = False,
        qk_rms_norm_cross: bool = False,
    ):
        super().__init__()
        # print(f"SparseStructureFlowModel: num_head_channels={num_head_channels}, share_mod={share_mod}, qk_rms_norm_cross={qk_rms_norm_cross}")
        # SparseStructureFlowModel: num_head_channels=64, share_mod=False, qk_rms_norm_cross=False

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
        assert use_fp16 == True
        assert share_mod == False
        assert qk_rms_norm == True
        assert qk_rms_norm_cross == False

        self.resolution = resolution
        self.in_channels = in_channels
        # self.num_heads = num_heads or model_channels // num_head_channels
        self.patch_size = patch_size
        self.share_mod = share_mod
        self.dtype = torch.float16 if use_fp16 else torch.float32

        self.t_embedder = TimestepEmbedder(model_channels)
        if share_mod: # False
            self.adaLN_modulation = nn.Sequential(
                nn.SiLU(),
                nn.Linear(model_channels, 6 * model_channels, bias=True)
            )

        if pe_mode == "ape": # True
            pos_embedder = AbsolutePositionEmbedder(model_channels, 3)
            coords = torch.meshgrid(*[torch.arange(res, device=self.device) for res in [resolution // patch_size] * 3], indexing='ij')
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
                share_mod=share_mod,
                qk_rms_norm=qk_rms_norm,
                qk_rms_norm_cross=qk_rms_norm_cross,
            )
            for _ in range(num_blocks)
        ])

        self.out_layer = nn.Linear(model_channels, out_channels * patch_size**3)

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
        self.blocks.apply(convert_module_to_f16)

    def convert_to_fp32(self) -> None:
        """
        Convert the torso of the model to float32.
        """
        self.blocks.apply(convert_module_to_f32)

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

        assert self.share_mod == False
        if self.share_mod: # False
            t_emb = self.adaLN_modulation(t_emb)

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
