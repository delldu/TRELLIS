from typing import *
import torch
import torch.nn as nn
from spconv.pytorch import SparseConvTensor
import todos
import pdb

__all__ = [
    'SparseTensor',
]

# https://zhuanlan.zhihu.com/p/720509689
class SparseTensor:
    def __init__(self, feats, coords, shape=None, scale=(1, 1, 1), spatial_cache={}):
        if shape is None: # False | True
            shape = self.__cal_shape(feats, coords)
        spatial_shape = (coords.max(0)[0][1:] + 1).tolist()

        self.data = SparseConvTensor(feats.reshape(feats.shape[0], -1), coords, spatial_shape, shape[0]) #  shape[0] -- batch_size
        self.data._features = feats # !!! import !!!

        self.shape = shape
        self.scale = scale
        self.spatial_cache = spatial_cache

    def __cal_shape(self, feats, coords):
        # (Pdb) coords.size() -- [14955, 4], feats.size() -- [14955, 8]
        shape = []
        shape.append(coords[:, 0].max().item() + 1)
        shape.extend([*feats.shape[1:]])
        # assert shape == [1, 8] or ...
        return torch.Size(shape) # [1, 8]
    
    @property
    def feats(self) -> torch.Tensor:
        return self.data.features
    
    @property
    def coords(self) -> torch.Tensor:
        return self.data.indices
        
    @property
    def dtype(self):
        return self.data.features.dtype

    @property
    def device(self):
        return self.data.features.device

    def output_var(self, s):
        todos.debug.output_var(f"{s} data.coords", self.data.indices)
        todos.debug.output_var(f"{s} data.features", self.data.features)


    def to(self, *args) -> 'SparseTensor':
        # ==> pdb.set_trace()
        dtype = None
        if len(args) == 1 and isinstance(args[0], torch.dtype):
            dtype = args[0]

        assert dtype == torch.float16
        new_feats = self.feats.to(dtype=dtype)
        return self.replace(new_feats, self.coords)

    def type(self, dtype):
        # ==> pdb.set_trace()
        new_feats = self.feats.type(dtype)
        return self.replace(new_feats, self.coords)

    def float(self) -> 'SparseTensor':
        # ==> pdb.set_trace()
        new_feats = self.feats.float()
        return self.replace(new_feats, self.coords)
    
    def reshape(self, *shape) -> 'SparseTensor':
        # ==> pdb.set_trace()
        new_feats = self.feats.reshape(self.feats.shape[0], *shape)
        return self.replace(new_feats, self.coords)
    
    def unbind(self, dim: int) -> List['SparseTensor']:
        # return sparse_unbind(self, dim)
        assert dim == 1
        feats = self.feats.unbind(dim)
        return [self.replace(f, self.coords) for f in feats]

    def replace(self, feats: torch.Tensor, coords: Optional[torch.Tensor] = None) -> 'SparseTensor':
        new_shape = [self.shape[0]]
        new_shape.extend(feats.shape[1:])
        new_coords = coords
        if coords is None:
            new_coords = self.coords

        new_tensor = SparseTensor(feats, new_coords, torch.Size(new_shape), self.scale, self.spatial_cache)

        return new_tensor


    def __elemwise__(self, other: Union[torch.Tensor, 'SparseTensor'], op: callable) -> 'SparseTensor':
        if isinstance(other, SparseTensor):
            other = other.feats
        new_feats = op(self.feats, other)
        new_tensor = self.replace(new_feats)

        return new_tensor

    def __add__(self, other: Union[torch.Tensor, 'SparseTensor', float]) -> 'SparseTensor':
        return self.__elemwise__(other, torch.add)

    def __sub__(self, other: Union[torch.Tensor, 'SparseTensor', float]) -> 'SparseTensor':
        return self.__elemwise__(other, torch.sub)

    def __mul__(self, other: Union[torch.Tensor, 'SparseTensor', float]) -> 'SparseTensor':
        return self.__elemwise__(other, torch.mul)

    def __rmul__(self, other: Union[torch.Tensor, 'SparseTensor', float]) -> 'SparseTensor':
        # ==> pdb.set_trace()
        return self.__elemwise__(other, torch.mul)

    def __getitem__(self, idx):
        # ==> pdb.set_trace()
        assert idx == 0
        return SparseTensor(self.feats, self.coords)

    def register_spatial_cache(self, key, value) -> None:
        scale_key = str(self.scale)
        # print(f"register_spatial_cache: scale_key = {scale_key}, key = {key}, value={value} ...")
        if scale_key not in self.spatial_cache:
            self.spatial_cache[scale_key] = {}
        self.spatial_cache[scale_key][key] = value

    def get_spatial_cache(self, key=None):
        scale_key = str(self.scale)
        cur_scale_cache = self.spatial_cache.get(scale_key, {})
        assert key is not None
        return cur_scale_cache.get(key, None)
