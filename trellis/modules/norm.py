import torch
import torch.nn as nn
from torch.nn import functional as F

class LayerNorm32(nn.LayerNorm):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # return super().forward(x.float()).type(x.dtype)
        return F.layer_norm(
            x.float(), self.normalized_shape, self.weight, self.bias, self.eps
        ).to(x.dtype)

    
# class ChannelLayerNorm32(LayerNorm32):
class ChannelLayerNorm32(nn.LayerNorm):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        DIM = x.dim()
        x = x.permute(0, *range(2, DIM), 1).contiguous()

        # x = super().forward(x)
        x = F.layer_norm(
            x.float(), self.normalized_shape, self.weight, self.bias, self.eps
        ).to(x.dtype)

        x = x.permute(0, DIM-1, *range(1, DIM-1)).contiguous()
        return x
    