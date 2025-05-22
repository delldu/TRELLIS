from typing import *
import torch
import torch.nn as nn
from ...modules.utils import convert_module_to_f16, convert_module_to_f32
from ...modules import sparse as sp
from ...modules.transformer import AbsolutePositionEmbedder
from ...modules.sparse.transformer import SparseTransformerBlock
import pdb

def block_attn_config(self):
    """
    Return the attention configuration of the model.
    """
    for i in range(self.num_blocks):
        if self.attn_mode == "shift_window":
            yield "serialized", self.window_size, 0, (16 * (i % 2),) * 3, sp.SerializeMode.Z_ORDER
        elif self.attn_mode == "shift_sequence":
            yield "serialized", self.window_size, self.window_size // 2 * (i % 2), (0, 0, 0), sp.SerializeMode.Z_ORDER
        elif self.attn_mode == "shift_order":
            yield "serialized", self.window_size, 0, (0, 0, 0), sp.SerializeModes[i % 4]
        elif self.attn_mode == "full":
            yield "full", None, None, None, None
        elif self.attn_mode == "swin":
            # attn_mode, window_size, shift_sequence, shift_window, serialize_mode
            yield "windowed", self.window_size, None, self.window_size // 2 * (i % 2), None


class SparseTransformerBase(nn.Module):
    """
    Sparse Transformer without output layers.
    Serve as the base class for encoder and decoder.
    """
    def __init__(
        self,
        in_channels: int,
        model_channels: int,
        num_blocks: int,
        num_heads: Optional[int] = None,
        num_head_channels: Optional[int] = 64,
        mlp_ratio: float = 4.0,
        attn_mode = "swin",
        window_size = 8,
        pe_mode = "ape",
        use_fp16: bool = True,
        qk_rms_norm: bool = False,
    ):
        super().__init__()
        # == SparseTransformerBase: attn_mode=swin, window_size=8

        # assert in_channels == 8
        # assert model_channels == 768
        assert num_blocks == 12
        # assert num_heads == 12
        assert num_head_channels == 64
        assert mlp_ratio == 4
        assert attn_mode == 'swin'
        assert window_size == 8
        assert pe_mode == 'ape'
        assert use_fp16 == True
        # assert qk_rms_norm == False

        self.num_blocks = num_blocks
        self.window_size = window_size

        assert num_heads is not None
        self.num_heads = num_heads or model_channels // num_head_channels
        self.attn_mode = attn_mode # 'swin'
        self.pe_mode = pe_mode # "ape"
        self.dtype = torch.float16 if use_fp16 else torch.float32

        if pe_mode == "ape": # True
            self.pos_embedder = AbsolutePositionEmbedder(model_channels)

        self.input_layer = sp.SparseLinear(in_channels, model_channels)
        self.blocks = nn.ModuleList([
            SparseTransformerBlock(
                model_channels,
                num_heads=self.num_heads,
                mlp_ratio=mlp_ratio,
                attn_mode=attn_mode,
                window_size=window_size,
                shift_sequence=shift_sequence,
                shift_window=shift_window,
                serialize_mode=serialize_mode,
                use_rope=(pe_mode == "rope"),
                qk_rms_norm=qk_rms_norm,
            )
            for attn_mode, window_size, shift_sequence, shift_window, serialize_mode in block_attn_config(self)
        ])

        # for attn_mode, window_size, shift_sequence, shift_window, serialize_mode in block_attn_config(self):
        #     print(attn_mode, window_size, shift_sequence, shift_window, serialize_mode)
        # windowed 8 None 0 None
        # windowed 8 None 4 None

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

    def forward(self, x: sp.SparseTensor) -> sp.SparseTensor:
        h2 = self.input_layer.float()(x)
        if self.pe_mode == "ape": # True
            h2 = h2 + self.pos_embedder(x.coords[:, 1:])
        h2 = h2.type(self.dtype)
        for block in self.blocks:
            h2 = block(h2)

        return h2
