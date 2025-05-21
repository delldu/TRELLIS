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
    '''
        coords=coords,
        feats=feats,    
    '''
    def __init__(self, *args, **kwargs):
        # args = ()
        # kwargs = {}
        method_id = 0

        if len(args) != 0:
            method_id = 0 if isinstance(args[0], torch.Tensor) else 1
        else:
            assert 'data' not in kwargs
            method_id = 1 if 'data' in kwargs else 0
            assert method_id == 0

        print(f"SparseTensor: method_id={method_id}, args={args}, kwargs={kwargs.keys()}")

        # assert method_id == 0 or 1
        if method_id == 0:
            feats, coords, shape, layout = args + (None,) * (4 - len(args)) # (None, None, None, None)
            # SparseTensor: args=(tensor([[ 7.917969,  8.007812,  0.144531,  ...,  ]]), tensor([[  0,   6,  60, 104], ...,
            #         [  0, 119,  67, 117]], dtype=torch.int32), torch.Size([1, 768])), kwargs=dict_keys([])
            # ----------------------------------------------------------------------------------------------
            # SparseTensor: args=(), kwargs=dict_keys(['feats', 'coords'])
            # ----------------------------------------------------------------------------------------------

            if 'feats' in kwargs: # True | False
                # ==> pdb.set_trace()
                feats = kwargs['feats']
                del kwargs['feats']
            if 'coords' in kwargs: # True | False
                coords = kwargs['coords']
                del kwargs['coords']

            if shape is None: # False | True
                shape = self.__cal_shape(feats, coords)
            else:
                pass #pdb.set_trace()

            spatial_shape = (coords.max(0)[0][1:] + 1).tolist() # xxxx_3333
            # self.data = SparseConvTensor(feats.reshape(feats.shape[0], -1), coords, spatial_shape, shape[0], **kwargs)
            # self.data._features = feats
            # pdb.set_trace()

            self.data = SparseConvTensor(feats, coords, spatial_shape, shape[0]) #  shape[0] -- batch_size

        elif method_id == 1: # SparseConvTensor
            # SparseTensor: args=(SparseConvTensor[shape=torch.Size([119640, 768])],), 
            #     kwargs=dict_keys(['shape', 'scale', 'spatial_cache'])
            # ----------------------------------------------------------------------------------------------
            data, shape, layout = args + (None,) * (3 - len(args))

            if 'shape' in kwargs: # True
                shape = kwargs['shape']
                del kwargs['shape']

            self.data = data
            if shape is None: # True
                shape = self.__cal_shape(self.feats, self.coords)
            else:
                pass # pdb.set_trace()

        self._shape = shape
        self._scale = kwargs.get('scale', (1, 1, 1))
        self._spatial_cache = kwargs.get('spatial_cache', {})

    def __cal_shape(self, feats, coords):
        # (Pdb) coords.size() -- [14955, 4], feats.size() -- [14955, 8]
        shape = []
        shape.append(coords[:, 0].max().item() + 1)
        shape.extend([*feats.shape[1:]])

        # assert shape == [1, 8] or ...
        return torch.Size(shape) # [1, 8]
    
    @property
    def shape(self) -> torch.Size:
        return self._shape

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


    def to(self, *args, **kwargs) -> 'SparseTensor':
        # ==> pdb.set_trace()
        # print(f"to ====> args={args}, kwargs={kwargs.keys()} ...")
        # to ====> args=(torch.float16,), kwargs=dict_keys([]) ...

        device = None
        dtype = None
        if len(args) == 2:
            pdb.set_trace()
            device, dtype = args
        elif len(args) == 1:
            if isinstance(args[0], torch.dtype):
                dtype = args[0]

        assert device == None
        assert dtype == torch.float16
        new_feats = self.feats.to(device=device, dtype=dtype)
        new_coords = self.coords.to(device=device)
        return self.replace(new_feats, new_coords)

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
        # -------------------------------------------------------------------------
        # features: torch.Tensor,
        # indices: torch.Tensor,
        # spatial_shape: Union[List[int], np.ndarray],
        # batch_size: int,
        # grid: Optional[torch.Tensor] = None,
        # voxel_num: Optional[torch.Tensor] = None,
        # indice_dict: Optional[dict] = None,
        # benchmark: bool = False,
        # permanent_thrust_allocator: bool = False,
        # enable_timer: bool = False,
        # force_algo: Optional[ConvAlgo] = None
        # -------------------------------------------------------------------------
        # new_data = SparseTensorData(
        #     self.data.features.reshape(self.data.features.shape[0], -1),
        #     self.data.indices,
        #     self.data.spatial_shape,
        #     self.data.batch_size,
        # )
        new_data = SparseConvTensor(
            self.data.features.reshape(self.data.features.shape[0], -1),
            self.data.indices,
            self.data.spatial_shape,
            self.data.batch_size,
        )

        assert self.data.batch_size == 1
        new_data._features = feats
        # new_data.benchmark = self.data.benchmark
        # new_data.benchmark_record = self.data.benchmark_record
        # new_data.thrust_allocator = self.data.thrust_allocator
        # new_data._timer = self.data._timer
        # new_data.force_algo = self.data.force_algo
        # new_data.int8_scale = self.data.int8_scale
        if coords is not None:
            # ==> pdb.set_trace()
            new_data.indices = coords
        else:
            pass #pdb.set_trace()

        # xxxx_3333
        new_tensor = SparseTensor(new_data, shape=torch.Size(new_shape), 
            scale=self._scale, 
            spatial_cache=self._spatial_cache,
        )
        return new_tensor

        # if coords is None:
        #     coords = self.data.indices
        # return SparseTensor(feats, coords)

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
        return SparseTensor(feats=self.feats, coords=self.coords)

    # xxxx_3333
    def register_spatial_cache(self, key, value) -> None:
        """ Register a spatial cache. """
        scale_key = str(self._scale)
        print(f"register_spatial_cache: scale_key = {scale_key}, key = {key}, value={value} ...")
        if scale_key not in self._spatial_cache:
            self._spatial_cache[scale_key] = {}
        self._spatial_cache[scale_key][key] = value

    # xxxx_3333
    def get_spatial_cache(self, key=None):
        """
        Get a spatial cache.
        """
        scale_key = str(self._scale)
        cur_scale_cache = self._spatial_cache.get(scale_key, {})
        assert key is not None
        return cur_scale_cache.get(key, None)
