from typing import *
import torch
import torch.nn as nn
import torch.nn.functional as F
# from ..modules.norm import GroupNorm32, ChannelLayerNorm32
from ..modules.norm import ChannelLayerNorm32
from ..modules.spatial import pixel_shuffle_3d
# from ..modules.utils import zero_module, convert_module_to_f16, convert_module_to_f32
from ..modules.utils import convert_module_to_f16, convert_module_to_f32
import todos
import pdb

def norm_layer(norm_type: str, *args, **kwargs) -> nn.Module:
    """
    Return a normalization layer.
    """
    # print(f"== norm_layer: norm_type={norm_type}, args={args}, kwargs={kwargs} ...")
    # == norm_layer: norm_type=layer, args=(512,), kwargs={} ...
    # == norm_layer: norm_type=layer, args=(512,), kwargs={} ...
    # == norm_layer: norm_type=layer, args=(512,), kwargs={} ...
    # == norm_layer: norm_type=layer, args=(512,), kwargs={} ...
    # == norm_layer: norm_type=layer, args=(512,), kwargs={} ...
    # == norm_layer: norm_type=layer, args=(512,), kwargs={} ...
    # == norm_layer: norm_type=layer, args=(512,), kwargs={} ...
    # == norm_layer: norm_type=layer, args=(512,), kwargs={} ...
    # == norm_layer: norm_type=layer, args=(128,), kwargs={} ...
    # == norm_layer: norm_type=layer, args=(128,), kwargs={} ...
    # == norm_layer: norm_type=layer, args=(128,), kwargs={} ...
    # == norm_layer: norm_type=layer, args=(128,), kwargs={} ...
    # == norm_layer: norm_type=layer, args=(32,), kwargs={} ...
    # == norm_layer: norm_type=layer, args=(32,), kwargs={} ...
    # == norm_layer: norm_type=layer, args=(32,), kwargs={} ...
    # == norm_layer: norm_type=layer, args=(32,), kwargs={} ...
    # == norm_layer: norm_type=layer, args=(32,), kwargs={} ...

    assert norm_type == "layer"

    # if norm_type == "group":
    #     return GroupNorm32(32, *args, **kwargs)
    # elif norm_type == "layer":
    #     return ChannelLayerNorm32(*args, **kwargs)
    # else:
    #     raise ValueError(f"Invalid norm type {norm_type}")
    return ChannelLayerNorm32(*args, **kwargs)


class ResBlock3d(nn.Module):
    def __init__(
        self,
        channels: int,
        out_channels: Optional[int] = None,
        norm_type: Literal["group", "layer"] = "layer",
    ):
        super().__init__()
        # print(f"== ResBlock3d: channels={channels}, out_channels={out_channels}, norm_type={norm_type} ...")
        # == ResBlock3d: channels=512, out_channels=512, norm_type=layer ...
        # == ResBlock3d: channels=512, out_channels=512, norm_type=layer ...
        # == ResBlock3d: channels=512, out_channels=512, norm_type=layer ...
        # == ResBlock3d: channels=128, out_channels=128, norm_type=layer ...
        # == ResBlock3d: channels=32, out_channels=32, norm_type=layer ...
        assert norm_type == "layer"

        # self.channels = channels
        # self.out_channels = out_channels or channels

        self.norm1 = norm_layer(norm_type, channels)
        self.norm2 = norm_layer(norm_type, out_channels)
        self.conv1 = nn.Conv3d(channels, out_channels, 3, padding=1)
        # self.conv2 = zero_module(nn.Conv3d(out_channels, out_channels, 3, padding=1))
        self.conv2 = nn.Conv3d(out_channels, out_channels, 3, padding=1)
        self.skip_connection = nn.Conv3d(channels, out_channels, 1) if channels != out_channels else nn.Identity()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h2 = self.norm1.float()(x.float())
        h2 = F.silu(h2)
        h2 = self.conv1(h2.type(x.dtype))
        h2 = self.norm2.float()(h2.float())
        h2 = F.silu(h2)
        h2 = self.conv2(h2.type(x.dtype))
        h2 = h2 + self.skip_connection(x)

        return h2.type(x.dtype)


# class DownsampleBlock3d(nn.Module):
#     def __init__(
#         self,
#         in_channels: int,
#         out_channels: int,
#         mode: Literal["conv", "avgpool"] = "conv",
#     ):
#         assert mode in ["conv", "avgpool"], f"Invalid mode {mode}"

#         super().__init__()
#         self.in_channels = in_channels
#         self.out_channels = out_channels

#         if mode == "conv":
#             self.conv = nn.Conv3d(in_channels, out_channels, 2, stride=2)
#         elif mode == "avgpool":
#             assert in_channels == out_channels, "Pooling mode requires in_channels to be equal to out_channels"

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         if hasattr(self, "conv"):
#             return self.conv(x)
#         else:
#             return F.avg_pool3d(x, 2)


class UpsampleBlock3d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        mode: Literal["conv", "nearest"] = "conv",
    ):
        assert mode in ["conv", "nearest"], f"Invalid mode {mode}"

        super().__init__()
        # print(f"== UpsampleBlock3d: in_channels={in_channels}, out_channels={out_channels}, mode={mode}")
        # == UpsampleBlock3d: in_channels=512, out_channels=128, mode=conv
        # == UpsampleBlock3d: in_channels=128, out_channels=32, mode=conv
        # assert mode == "conv"
        # # self.in_channels = in_channels
        # # self.out_channels = out_channels
        # if mode == "conv":
        #     self.conv = nn.Conv3d(in_channels, out_channels*8, 3, padding=1)
        # elif mode == "nearest":
        #     assert in_channels == out_channels, "Nearest mode requires in_channels to be equal to out_channels"
        self.conv = nn.Conv3d(in_channels, out_channels*8, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # if hasattr(self, "conv"):
        #     x = self.conv(x)
        #     return pixel_shuffle_3d(x, 2)
        # else:
        #     return F.interpolate(x, scale_factor=2, mode="nearest")
        x = self.conv(x)
        return pixel_shuffle_3d(x, 2)
        

# class SparseStructureEncoder(nn.Module):
#     """
#     Encoder for Sparse Structure (\mathcal{E}_S in the paper Sec. 3.3).
    
#     Args:
#         in_channels (int): Channels of the input.
#         latent_channels (int): Channels of the latent representation.
#         num_res_blocks (int): Number of residual blocks at each resolution.
#         channels (List[int]): Channels of the encoder blocks.
#         num_res_blocks_middle (int): Number of residual blocks in the middle.
#         norm_type (Literal["group", "layer"]): Type of normalization layer.
#         use_fp16 (bool): Whether to use FP16.
#     """
#     def __init__(
#         self,
#         in_channels: int,
#         latent_channels: int,
#         num_res_blocks: int,
#         channels: List[int],
#         num_res_blocks_middle: int = 2,
#         norm_type: Literal["group", "layer"] = "layer",
#         use_fp16: bool = False,
#     ):
#         super().__init__()
#         self.in_channels = in_channels
#         # self.latent_channels = latent_channels
#         # self.num_res_blocks = num_res_blocks
#         # self.channels = channels
#         # self.num_res_blocks_middle = num_res_blocks_middle
#         self.norm_type = norm_type
#         self.use_fp16 = use_fp16
#         self.dtype = torch.float16 if use_fp16 else torch.float32

#         self.input_layer = nn.Conv3d(in_channels, channels[0], 3, padding=1)

#         self.blocks = nn.ModuleList([])
#         for i, ch in enumerate(channels):
#             self.blocks.extend([
#                 ResBlock3d(ch, ch)
#                 for _ in range(num_res_blocks)
#             ])
#             if i < len(channels) - 1:
#                 self.blocks.append(
#                     DownsampleBlock3d(ch, channels[i+1])
#                 )
        
#         self.middle_block = nn.Sequential(*[
#             ResBlock3d(channels[-1], channels[-1])
#             for _ in range(num_res_blocks_middle)
#         ])

#         self.out_layer = nn.Sequential(
#             norm_layer(norm_type, channels[-1]),
#             nn.SiLU(),
#             nn.Conv3d(channels[-1], latent_channels*2, 3, padding=1)
#         )

#         if use_fp16:
#             self.convert_to_fp16()
#         pdb.set_trace()

#     @property
#     def device(self) -> torch.device:
#         """
#         Return the device of the model.
#         """
#         return next(self.parameters()).device

#     def convert_to_fp16(self) -> None:
#         """
#         Convert the torso of the model to float16.
#         """
#         self.use_fp16 = True
#         self.dtype = torch.float16
#         self.blocks.apply(convert_module_to_f16)
#         self.middle_block.apply(convert_module_to_f16)

#     def convert_to_fp32(self) -> None:
#         """
#         Convert the torso of the model to float32.
#         """
#         self.use_fp16 = False
#         self.dtype = torch.float32
#         self.blocks.apply(convert_module_to_f32)
#         self.middle_block.apply(convert_module_to_f32)

#     def forward(self, x: torch.Tensor, sample_posterior: bool = False, return_raw: bool = False) -> torch.Tensor:
#         h = self.input_layer(x)
#         h = h.type(self.dtype)

#         for block in self.blocks:
#             h = block(h)
#         h = self.middle_block(h)

#         h = h.type(x.dtype)
#         h = self.out_layer(h)

#         mean, logvar = h.chunk(2, dim=1)

#         if sample_posterior:
#             std = torch.exp(0.5 * logvar)
#             z = mean + std * torch.randn_like(std)
#         else:
#             z = mean
            
#         if return_raw:
#             return z, mean, logvar
#         return z

class SparseStructureDecoder(nn.Module):
    """
    "name": "SparseStructureDecoder",
    "args": {
        "out_channels": 1,
        "latent_channels": 8,
        "num_res_blocks": 2,
        "num_res_blocks_middle": 2,
        "channels": [512, 128, 32],
        "use_fp16": true
    }
    """ 
    def __init__(
        self,
        out_channels: int,
        latent_channels: int,
        num_res_blocks: int,
        channels: List[int],
        num_res_blocks_middle: int = 2,
        norm_type: Literal["group", "layer"] = "layer",
        use_fp16: bool = False,
    ):
        super().__init__()
        # print(f"SparseStructureDecoder: norm_type={norm_type}")
        # SparseStructureDecoder: norm_type=layer

        assert out_channels == 1
        assert latent_channels == 8
        assert num_res_blocks == 2
        assert channels == [512, 128, 32]
        assert num_res_blocks_middle == 2
        assert norm_type == 'layer'
        assert use_fp16 == True

        self.norm_type = norm_type
        self.use_fp16 = use_fp16
        self.dtype = torch.float16 if use_fp16 else torch.float32

        self.input_layer = nn.Conv3d(latent_channels, channels[0], 3, padding=1)

        self.middle_block = nn.Sequential(*[
            ResBlock3d(channels[0], channels[0])
            for _ in range(num_res_blocks_middle)
        ])

        self.blocks = nn.ModuleList([])
        for i, ch in enumerate(channels):
            self.blocks.extend([
                ResBlock3d(ch, ch)
                for _ in range(num_res_blocks)
            ])
            if i < len(channels) - 1:
                self.blocks.append(
                    UpsampleBlock3d(ch, channels[i+1])
                )

        self.out_layer = nn.Sequential(
            norm_layer(norm_type, channels[-1]),
            nn.SiLU(),
            nn.Conv3d(channels[-1], out_channels, 3, padding=1)
        )

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
        self.use_fp16 = True
        self.dtype = torch.float16
        self.blocks.apply(convert_module_to_f16)
        self.middle_block.apply(convert_module_to_f16)

    def convert_to_fp32(self) -> None:
        """
        Convert the torso of the model to float32.
        """
        self.use_fp16 = False
        self.dtype = torch.float32
        self.blocks.apply(convert_module_to_f32)
        self.middle_block.apply(convert_module_to_f32)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # tensor [x] size: [1, 8, 16, 16, 16], min: -5.693636, max: 4.097641, mean: 0.010038

        h2 = self.input_layer(x.type(self.dtype))
        
        h2 = h2.type(self.dtype)
        h2 = self.middle_block(h2) # self.middle_block -- device='cuda:0', dtype=torch.float16

        for block in self.blocks:
            h2 = block(h2)

        h2 = h2.type(x.dtype)
        h2 = self.out_layer.float()(h2.float())
        # tensor [h2] size: [1, 1, 64, 64, 64], min: -216.489441, max: 181.025513, mean: -145.066437

        return h2.type(x.dtype)
