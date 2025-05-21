from typing import *
import torch
import torch.nn as nn
from . import SparseTensor
import todos
import pdb

__all__ = [
    'SparseDownsample',
    'SparseUpsample',
    'SparseSubdivide'
]


class SparseDownsample(nn.Module):
    """
    Downsample a sparse tensor by a factor of `factor`.
    Implemented as average pooling.
    """
    def __init__(self, factor: int):
        super().__init__()
        assert factor == 2
        self.factor = factor

    def forward(self, input: SparseTensor) -> SparseTensor:
        # tensor [input.coords] size: [14955, 4], min: 0.0, max: 63.0, mean: 23.262018
        # tensor [input.feats] size: [14955, 128], min: -14.195312, max: 6.210938, mean: -0.022306

        DIM = input.coords.shape[-1] - 1
        factor = (self.factor,) * DIM
        assert DIM == len(factor), 'Input coordinates must have the same dimension as the downsample factor.'

        # pdb.set_trace()

        coord = list(input.coords.unbind(dim=-1))
        for i, f in enumerate(factor):
            coord[i+1] = coord[i+1] // f

        MAX = [coord[i+1].max().item() + 1 for i in range(DIM)] # DIM == 3
        OFFSET = torch.cumprod(torch.tensor(MAX[::-1]), 0).tolist()[::-1] + [1]
        # OFFSET -- [22080, 736, 32, 1]
        code = sum([c * o for c, o in zip(coord, OFFSET)])
        # MAX is list: len = 3, [30, 23, 32]
        # tensor [code] size: [14955], min: 1242.0, max: 21982.0, mean: 11982.15332

        code, idx = code.unique(return_inverse=True)
        # tensor [code] size: [3185], min: 1242.0, max: 21982.0, mean: 11973.475586
        # tensor [idx] size: [14955], min: 0.0, max: 3184.0, mean: 1589.802124

        new_feats = torch.scatter_reduce(
            torch.zeros(code.shape[0], input.feats.shape[1], device=input.feats.device, dtype=input.feats.dtype),
            dim=0,
            index=idx.unsqueeze(1).expand(-1, input.feats.shape[1]),
            src=input.feats,
            reduce='mean'
        )
        new_coords = torch.stack(
            [code // OFFSET[0]] +
            [(code // OFFSET[i+1]) % MAX[i] for i in range(DIM)],
            dim=-1
        )
        # xxxx_3333

        # tensor [new_feats] size: [3185, 128], min: -7.777344, max: 2.693359, mean: -0.022208
        # tensor [new_coords] size: [3185, 4], min: 0.0, max: 31.0, mean: 11.572449
        out = SparseTensor(new_feats, new_coords, input.shape,)
        out._scale = tuple([s // f for s, f in zip(input._scale, factor)])
        out._spatial_cache = input._spatial_cache

        # xxxx_3333
        # tensor [input.coords] size: [14955, 4], min: 0.0, max: 63.0, mean: 23.262018

        out.register_spatial_cache(f'upsample_{factor}_coords', input.coords)
        # out.register_spatial_cache(f'upsample_{factor}_layout', input.layout)
        out.register_spatial_cache(f'upsample_{factor}_idx', idx)

        return out


class SparseUpsample(nn.Module):
    """
    Upsample a sparse tensor by a factor of `factor`.
    Implemented as nearest neighbor interpolation.
    """
    def __init__(self, factor: int):
        super().__init__()
        assert factor == 2
        # self.factor = tuple(factor) if isinstance(factor, (list, tuple)) else factor
        self.factor = factor

    def forward(self, input: SparseTensor) -> SparseTensor:
        # tensor [input.feats] size: [3185, 2048], min: -600.0, max: 91.5, mean: -0.028333
        # tensor [input.coords] size: [3185, 4], min: 0.0, max: 31.0, mean: 11.572449

        DIM = input.coords.shape[-1] - 1 # === 3
        # factor = self.factor if isinstance(self.factor, tuple) else (self.factor,) * DIM
        factor = (self.factor,) * DIM
        # assert DIM == len(factor), 'Input coordinates must have the same dimension as the upsample factor.'

        # xxxx_3333
        new_coords = input.get_spatial_cache(f'upsample_{factor}_coords')
        # new_layout = input.get_spatial_cache(f'upsample_{factor}_layout')
        idx = input.get_spatial_cache(f'upsample_{factor}_idx') # idx.size() -- [14955]

        # if any([x is None for x in [new_coords, new_layout, idx]]):
        #     raise ValueError('Upsample cache not found. SparseUpsample must be paired with SparseDownsample.')
        # else:
        #     # print(f" ................. upsample_{factor}_* SparseUpsample cached OK .................")
        #     assert factor == (2, 2, 2)

        new_feats = input.feats[idx]
        # xxxx_3333
        # tensor [new_coords] size: [14955, 4], min: 0.0, max: 63.0, mean: 23.262018
        # tensor [new_feats] size: [14955, 2048], min: -600.0, max: 91.5, mean: -0.087899
        # (Pdb) input.shape, new_layout
        # (torch.Size([1, 2048]), [slice(0, 14955, None)])

        # out = SparseTensor(new_feats, new_coords, input.shape, new_layout)
        out = SparseTensor(new_feats, new_coords, input.shape)
        out._scale = tuple([s * f for s, f in zip(input._scale, factor)])
        out._spatial_cache = input._spatial_cache
        # out._spatial_cache['(0, 0, 0)'].keys() -- dict_keys(['upsample_(2, 2, 2)_coords', 'upsample_(2, 2, 2)_layout', 'upsample_(2, 2, 2)_idx'])

        return out
    
class SparseSubdivide(nn.Module):
    """
    Upsample a sparse tensor by a factor of `factor`.
    Implemented as nearest neighbor interpolation.
    """
    def __init__(self):
        super().__init__()

    def forward(self, input: SparseTensor) -> SparseTensor:
        # tensor [input.coords] size: [14955, 4], min: 0.0, max: 63.0, mean: 23.262018
        # tensor [input.feats] size: [14955, 768], min: -0.278465, max: 8.002379, mean: 0.094177

        DIM = input.coords.shape[-1] - 1 # === 3
        # upsample scale=2^DIM
        n_cube = torch.ones([2] * DIM, device=input.device, dtype=torch.int) # size() -- [2, 2, 2]
        n_coords = torch.nonzero(n_cube) # size() -- [8, 3]
        n_coords = torch.cat([torch.zeros_like(n_coords[:, :1]), n_coords], dim=-1) # size() -- [8, 4]
        factor = n_coords.shape[0] # ====> 8
        assert factor == 2 ** DIM

        # print(n_coords.shape)
        new_coords = input.coords.clone()
        new_coords[:, 1:] *= 2
        new_coords = new_coords.unsqueeze(1) + n_coords.unsqueeze(0).to(new_coords.dtype)
        # tensor [new_coords] size: [14955, 8, 4], min: 0.0, max: 127.0, mean: 46.899036
    
        new_feats = input.feats.unsqueeze(1).expand(input.feats.shape[0], factor, *input.feats.shape[1:])
        # tensor [new_feats] size: [14955, 8, 768], min: -0.278465, max: 8.002379, mean: 0.094177

        out = SparseTensor(new_feats.flatten(0, 1), new_coords.flatten(0, 1), input.shape)
        out._scale = input._scale * 2
        out._spatial_cache = input._spatial_cache

        return out

