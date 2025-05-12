import os

from typing import *
import torch
import torch.nn as nn
import torch.nn.functional as F
import todos
import pdb


def pixel_shuffle_3d(x: torch.Tensor, scale_factor: int) -> torch.Tensor:
    """
    3D pixel shuffle.
    """
    B, C, H, W, D = x.shape
    C_ = C // scale_factor**3
    x = x.reshape(B, C_, scale_factor, scale_factor, scale_factor, H, W, D)
    x = x.permute(0, 1, 5, 2, 6, 3, 7, 4)
    x = x.reshape(B, C_, H*scale_factor, W*scale_factor, D*scale_factor)
    return x


class LayerNorm32(nn.LayerNorm):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return super().forward(x.float()).type(x.dtype)
    

# class GroupNorm32(nn.GroupNorm):
#     """
#     A GroupNorm layer that converts to float32 before the forward pass.
#     """
#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         return super().forward(x.float()).type(x.dtype)
    
# xxxx_9999    
class ChannelLayerNorm32(LayerNorm32):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        DIM = x.dim()
        x = x.permute(0, *range(2, DIM), 1).contiguous()
        x = super().forward(x)
        x = x.permute(0, DIM-1, *range(1, DIM-1)).contiguous()
        return x

def norm_layer(norm_type: str, *args, **kwargs) -> nn.Module:
    """
    Return a normalization layer.
    """
    # print(f"== norm_layer: norm_type={norm_type}, args={args}, kwargs={kwargs} ...")
    # == norm_layer: norm_type=layer, args=(512,), kwargs={} ...
    # == norm_layer: norm_type=layer, args=(128,), kwargs={} ...
    # == norm_layer: norm_type=layer, args=(32,), kwargs={} ...

    assert norm_type == "layer"
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


class UpsampleBlock3d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        mode: Literal["conv", "nearest"] = "conv",
    ):
        assert mode in ["conv", "nearest"], f"Invalid mode {mode}"

        super().__init__()
        self.conv = nn.Conv3d(in_channels, out_channels*8, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        return pixel_shuffle_3d(x, 2)
        
# xxxx_1111
class SparseStructureDecoder(nn.Module):
    """
    "name": "SparseStructureDecoder",
    "args": {
        "out_channels": 1,
        "latent_channels": 8,
        "num_res_blocks": 2,
        "num_res_blocks_middle": 2,
        "channels": [512, 128, 32],
    }
    """ 
    def __init__(
        self,
        out_channels = 1,
        latent_channels = 8,
        num_res_blocks = 2,
        channels = [512, 128, 32],
        num_res_blocks_middle = 2,
        norm_type = "layer",
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

        self.norm_type = norm_type
        self.dtype = torch.float16
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

    def load_weights(self, model_path="models/image3d_dinov2.pth"):
        cdir = os.path.dirname(__file__)
        checkpoint = model_path if cdir == "" else cdir + "/" + model_path
        print(f"Loading {checkpoint} ...")
        self.load_state_dict(torch.load(checkpoint), strict=True)

    
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


if __name__ == "__main__":
    model = SparseStructureDecoder()
    model.eval()
    
    print(model)