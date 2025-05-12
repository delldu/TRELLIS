from typing import *
import torch
import torch.nn as nn
SparseTensorData = None # Lazy import

import todos
import pdb

# https://zhuanlan.zhihu.com/p/720509689
class SparseTensor:
    def __init__(self, *args, **kwargs):
        # args = ()
        # kwargs = {}
        # Lazy import of sparse tensor backend
        global SparseTensorData
        # SparseTensorData -- <class 'spconv.pytorch.core.SparseConvTensor'>
        if SparseTensorData is None: # False
            import importlib
            SparseTensorData = importlib.import_module('spconv.pytorch').SparseConvTensor
                
        method_id = 0
        if len(args) != 0:
            method_id = 0 if isinstance(args[0], torch.Tensor) else 1
        else:
            method_id = 1 if 'data' in kwargs else 0

        # assert method_id == 0 or 1
        if method_id == 0:
            feats, coords, shape, layout = args + (None,) * (4 - len(args)) # (None, None, None, None)

            # print(f"SparseTensor: method_id==0: kwargs={kwargs.keys()}, shape={shape}, layout={layout} ...")
            # SparseTensor: method_id==0: kwargs=dict_keys(['feats', 'coords']), shape=None, layout=None ...
            # SparseTensor: method_id==0: kwargs=dict_keys([]), shape=torch.Size([1, 192]), layout=None ...

            if 'feats' in kwargs: # True | False
                feats = kwargs['feats']
                del kwargs['feats']
            if 'coords' in kwargs: # True | False
                coords = kwargs['coords']
                del kwargs['coords']

            if 'shape' in kwargs: # False
                pdb.set_trace()
                shape = kwargs['shape']
                del kwargs['shape']
            if 'layout' in kwargs: # False
                pdb.set_trace()
                layout = kwargs['layout']
                del kwargs['layout']


            if shape is None: # False | True
                shape = self.__cal_shape(feats, coords)
            if layout is None: # True
                layout = self.__cal_layout(coords, shape[0])

            spatial_shape = list(coords.max(0)[0] + 1)[1:]
            # spatial_shape -- [tensor(60, device='cuda:0', dtype=torch.int32), 
            #     tensor(46, device='cuda:0', dtype=torch.int32), 
            #     tensor(64, device='cuda:0', dtype=torch.int32)]
            self.data = SparseTensorData(feats.reshape(feats.shape[0], -1), coords, spatial_shape, shape[0], **kwargs)
            self.data._features = feats
        elif method_id == 1:
            data, shape, layout = args + (None,) * (3 - len(args))
            # print(f"SparseTensor: method_id == 1: kwargs={kwargs.keys()}, shape={shape}, layout={layout} ...")
            # SparseTensor: method_id == 1: kwargs=dict_keys(['shape', 'layout', 'scale', 'spatial_cache']), shape=None, layout=None ...

            if 'data' in kwargs: # False
                pdb.set_trace()
                data = kwargs['data']
                del kwargs['data']
            if 'shape' in kwargs: # True
                shape = kwargs['shape']
                del kwargs['shape']
            if 'layout' in kwargs: # True
                layout = kwargs['layout']
                del kwargs['layout']

            self.data = data
            if shape is None: # True
                shape = self.__cal_shape(self.feats, self.coords)
            if layout is None: # False
                pdb.set_trace()
                layout = self.__cal_layout(self.coords, shape[0])

        self._shape = shape
        self._layout = layout
        self._scale = kwargs.get('scale', (1, 1, 1))
        self._spatial_cache = kwargs.get('spatial_cache', {})
        #pdb.set_trace()

    def __cal_shape(self, feats, coords):
        shape = []
        shape.append(coords[:, 0].max().item() + 1)
        shape.extend([*feats.shape[1:]])
        return torch.Size(shape)
    
    def __cal_layout(self, coords, batch_size):
        seq_len = torch.bincount(coords[:, 0], minlength=batch_size)
        offset = torch.cumsum(seq_len, dim=0) 
        layout = [slice((offset[i] - seq_len[i]).item(), offset[i].item()) for i in range(batch_size)]
        return layout
    
    @property
    def shape(self) -> torch.Size:
        return self._shape
    
    def dim(self) -> int:
        return len(self.shape)
    
    @property
    def layout(self) -> List[slice]:
        return self._layout

    @property
    def feats(self) -> torch.Tensor:
        return self.data.features
    
    @feats.setter
    def feats(self, value: torch.Tensor):
        self.data.features = value

    @property
    def coords(self) -> torch.Tensor:
        return self.data.indices
        
    @coords.setter
    def coords(self, value: torch.Tensor):
        self.data.indices = value

    @property
    def dtype(self):
        return self.feats.dtype

    @property
    def device(self):
        return self.feats.device

    def output_var(self, s):
        todos.debug.output_var(f"{s} data.coords", self.data.indices)
        todos.debug.output_var(f"{s} data.features", self.data.features)

    # @overload
    # def to(self, dtype: torch.dtype) -> 'SparseTensor': ...

    # @overload
    # def to(self, device: Optional[Union[str, torch.device]] = None, dtype: Optional[torch.dtype] = None) -> 'SparseTensor': ...

    def to(self, *args, **kwargs) -> 'SparseTensor':
        # ==> pdb.set_trace()
        device = None
        dtype = None
        if len(args) == 2:
            device, dtype = args
        elif len(args) == 1:
            if isinstance(args[0], torch.dtype):
                dtype = args[0]
            else:
                device = args[0]
        if 'dtype' in kwargs:
            assert dtype is None, "to() received multiple values for argument 'dtype'"
            dtype = kwargs['dtype']
        if 'device' in kwargs:
            assert device is None, "to() received multiple values for argument 'device'"
            device = kwargs['device']
        
        new_feats = self.feats.to(device=device, dtype=dtype)
        new_coords = self.coords.to(device=device)
        return self.replace(new_feats, new_coords)

    def type(self, dtype):
        new_feats = self.feats.type(dtype)
        return self.replace(new_feats)

    def cpu(self) -> 'SparseTensor':
        new_feats = self.feats.cpu()
        new_coords = self.coords.cpu()
        return self.replace(new_feats, new_coords)
    
    def cuda(self) -> 'SparseTensor':
        new_feats = self.feats.cuda()
        new_coords = self.coords.cuda()
        return self.replace(new_feats, new_coords)

    def half(self) -> 'SparseTensor':
        new_feats = self.feats.half()
        return self.replace(new_feats)
    
    def float(self) -> 'SparseTensor':
        new_feats = self.feats.float()
        return self.replace(new_feats)
    
    def reshape(self, *shape) -> 'SparseTensor':
        new_feats = self.feats.reshape(self.feats.shape[0], *shape)
        return self.replace(new_feats)
    
    def unbind(self, dim: int) -> List['SparseTensor']:
        return sparse_unbind(self, dim)

    def replace(self, feats: torch.Tensor, coords: Optional[torch.Tensor] = None) -> 'SparseTensor':
        new_shape = [self.shape[0]]
        new_shape.extend(feats.shape[1:])
        new_data = SparseTensorData(
            self.data.features.reshape(self.data.features.shape[0], -1),
            self.data.indices,
            self.data.spatial_shape,
            self.data.batch_size,
            self.data.grid,
            self.data.voxel_num,
            self.data.indice_dict
        )
        new_data._features = feats
        new_data.benchmark = self.data.benchmark
        new_data.benchmark_record = self.data.benchmark_record
        new_data.thrust_allocator = self.data.thrust_allocator
        new_data._timer = self.data._timer
        new_data.force_algo = self.data.force_algo
        new_data.int8_scale = self.data.int8_scale
        if coords is not None:
            new_data.indices = coords
        new_tensor = SparseTensor(new_data, shape=torch.Size(new_shape), layout=self.layout, scale=self._scale, \
                spatial_cache=self._spatial_cache)
        return new_tensor

    def __elemwise__(self, other: Union[torch.Tensor, 'SparseTensor'], op: callable) -> 'SparseTensor':
        # ==> pdb.set_trace()
        if isinstance(other, torch.Tensor): # True
            try:
                # print("===> sparse_batch_broadcast S1 ...")
                other = torch.broadcast_to(other, self.shape)
                # print("===> sparse_batch_broadcast S2 ...")
                other = sparse_batch_broadcast(self, other)
                # print("===> sparse_batch_broadcast S3 ...")
            except:
                # print(f"===> sparse_batch_broadcast S4 ... {type(other)}")
                # pdb.set_trace()
                pass
        if isinstance(other, SparseTensor):
            other = other.feats
        new_feats = op(self.feats, other)
        new_tensor = self.replace(new_feats)
        if isinstance(other, SparseTensor): # False
            pdb.set_trace()
            new_tensor._spatial_cache = self.__merge_sparse_cache(other)
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
        if isinstance(idx, int):
            idx = [idx]
        elif isinstance(idx, slice):
            idx = range(*idx.indices(self.shape[0]))
        elif isinstance(idx, torch.Tensor):
            if idx.dtype == torch.bool:
                assert idx.shape == (self.shape[0],), f"Invalid index shape: {idx.shape}"
                idx = idx.nonzero().squeeze(1)
            elif idx.dtype in [torch.int32, torch.int64]:
                assert len(idx.shape) == 1, f"Invalid index shape: {idx.shape}"
            else:
                raise ValueError(f"Unknown index type: {idx.dtype}")
        else:
            raise ValueError(f"Unknown index type: {type(idx)}")
        
        coords = []
        feats = []
        for new_idx, old_idx in enumerate(idx):
            coords.append(self.coords[self.layout[old_idx]].clone())
            coords[-1][:, 0] = new_idx
            feats.append(self.feats[self.layout[old_idx]])
        coords = torch.cat(coords, dim=0).contiguous()
        feats = torch.cat(feats, dim=0).contiguous()
        return SparseTensor(feats=feats, coords=coords)

    def register_spatial_cache(self, key, value) -> None:
        """
        Register a spatial cache.
        The spatial cache can be any thing you want to cache.
        The registery and retrieval of the cache is based on current scale.
        """
        scale_key = str(self._scale)
        if scale_key not in self._spatial_cache:
            self._spatial_cache[scale_key] = {}
        self._spatial_cache[scale_key][key] = value

    def get_spatial_cache(self, key=None):
        """
        Get a spatial cache.
        """
        scale_key = str(self._scale)
        cur_scale_cache = self._spatial_cache.get(scale_key, {})
        if key is None:
            return cur_scale_cache
        return cur_scale_cache.get(key, None)


def sparse_batch_broadcast(input: SparseTensor, other: torch.Tensor) -> torch.Tensor:
    """
    Broadcast a 1D tensor to a sparse tensor along the batch dimension then perform an operation.
    
    Args:
        input (torch.Tensor): 1D tensor to broadcast.
        target (SparseTensor): Sparse tensor to broadcast to.
        op (callable): Operation to perform after broadcasting. Defaults to torch.add.
    """
    coords, feats = input.coords, input.feats
    broadcasted = torch.zeros_like(feats)
    for k in range(input.shape[0]):
        broadcasted[input.layout[k]] = other[k]

    return broadcasted


def sparse_unbind(input: SparseTensor, dim: int) -> List[SparseTensor]:
    # ==> pdb.set_trace()
    if dim == 0:
        return [input[i] for i in range(input.shape[0])]
    else:
        feats = input.feats.unbind(dim)
        return [input.replace(f) for f in feats]


class SparseLinear(nn.Linear):
    def __init__(self, in_features, out_features, bias=True):
        super(SparseLinear, self).__init__(in_features, out_features, bias)

    def forward(self, input: SparseTensor) -> SparseTensor:
        # pdb.set_trace()
        return input.replace(super().forward(input.feats))

class SparseSiLU(nn.SiLU):
    def forward(self, input: SparseTensor) -> SparseTensor:
        return input.replace(super().forward(input.feats))


class SparseGELU(nn.GELU):
    def forward(self, input: SparseTensor) -> SparseTensor:
        return input.replace(super().forward(input.feats))



class SparseGroupNorm(nn.GroupNorm):
    def __init__(self, num_groups, num_channels, eps=1e-5, affine=True):
        super(SparseGroupNorm, self).__init__(num_groups, num_channels, eps, affine)

    def forward(self, input: SparseTensor) -> SparseTensor:
        nfeats = torch.zeros_like(input.feats)
        for k in range(input.shape[0]):
            bfeats = input.feats[input.layout[k]]
            bfeats = bfeats.permute(1, 0).reshape(1, input.shape[1], -1)
            bfeats = super().forward(bfeats)
            bfeats = bfeats.reshape(input.shape[1], -1).permute(1, 0)
            nfeats[input.layout[k]] = bfeats
        return input.replace(nfeats)


class SparseLayerNorm(nn.LayerNorm):
    def __init__(self, normalized_shape, eps=1e-5, elementwise_affine=True):
        super(SparseLayerNorm, self).__init__(normalized_shape, eps, elementwise_affine)

    def forward(self, input: SparseTensor) -> SparseTensor:
        nfeats = torch.zeros_like(input.feats)
        for k in range(input.shape[0]):
            bfeats = input.feats[input.layout[k]]
            bfeats = bfeats.permute(1, 0).reshape(1, input.shape[1], -1)
            bfeats = super().forward(bfeats)
            bfeats = bfeats.reshape(input.shape[1], -1).permute(1, 0)
            nfeats[input.layout[k]] = bfeats
        return input.replace(nfeats)


class SparseGroupNorm32(SparseGroupNorm):
    """
    A GroupNorm layer that converts to float32 before the forward pass.
    """
    def forward(self, x: SparseTensor) -> SparseTensor:
        return super().forward(x.float()).type(x.dtype)

class SparseLayerNorm32(SparseLayerNorm):
    """
    A LayerNorm layer that converts to float32 before the forward pass.
    """
    def forward(self, x: SparseTensor) -> SparseTensor:
        return super().forward(x.float()).type(x.dtype)

class SparseDownsample(nn.Module):
    """
    Downsample a sparse tensor by a factor of `factor`.
    Implemented as average pooling.
    """
    def __init__(self, factor: Union[int, Tuple[int, ...], List[int]]):
        super(SparseDownsample, self).__init__()
        self.factor = tuple(factor) if isinstance(factor, (list, tuple)) else factor

    def forward(self, input: SparseTensor) -> SparseTensor:
        DIM = input.coords.shape[-1] - 1
        factor = self.factor if isinstance(self.factor, tuple) else (self.factor,) * DIM
        assert DIM == len(factor), 'Input coordinates must have the same dimension as the downsample factor.'

        coord = list(input.coords.unbind(dim=-1))
        for i, f in enumerate(factor):
            coord[i+1] = coord[i+1] // f

        MAX = [coord[i+1].max().item() + 1 for i in range(DIM)]
        OFFSET = torch.cumprod(torch.tensor(MAX[::-1]), 0).tolist()[::-1] + [1]
        code = sum([c * o for c, o in zip(coord, OFFSET)])
        code, idx = code.unique(return_inverse=True)

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
        out = SparseTensor(new_feats, new_coords, input.shape,)
        out._scale = tuple([s // f for s, f in zip(input._scale, factor)])
        out._spatial_cache = input._spatial_cache

        out.register_spatial_cache(f'upsample_{factor}_coords', input.coords)
        out.register_spatial_cache(f'upsample_{factor}_layout', input.layout)
        out.register_spatial_cache(f'upsample_{factor}_idx', idx)

        return out


class SparseUpsample(nn.Module):
    """
    Upsample a sparse tensor by a factor of `factor`.
    Implemented as nearest neighbor interpolation.
    """
    def __init__(self, factor: Union[int, Tuple[int, int, int], List[int]]):
        super(SparseUpsample, self).__init__()
        self.factor = tuple(factor) if isinstance(factor, (list, tuple)) else factor

    def forward(self, input: SparseTensor) -> SparseTensor:
        DIM = input.coords.shape[-1] - 1
        factor = self.factor if isinstance(self.factor, tuple) else (self.factor,) * DIM
        assert DIM == len(factor), 'Input coordinates must have the same dimension as the upsample factor.'

        new_coords = input.get_spatial_cache(f'upsample_{factor}_coords')
        new_layout = input.get_spatial_cache(f'upsample_{factor}_layout')
        idx = input.get_spatial_cache(f'upsample_{factor}_idx')
        if any([x is None for x in [new_coords, new_layout, idx]]):
            raise ValueError('Upsample cache not found. SparseUpsample must be paired with SparseDownsample.')
        new_feats = input.feats[idx]
        out = SparseTensor(new_feats, new_coords, input.shape, new_layout)
        out._scale = tuple([s * f for s, f in zip(input._scale, factor)])
        out._spatial_cache = input._spatial_cache
        return out
    
class SparseSubdivide(nn.Module):
    """
    Upsample a sparse tensor by a factor of `factor`.
    Implemented as nearest neighbor interpolation.
    """
    def __init__(self):
        super(SparseSubdivide, self).__init__()

    def forward(self, input: SparseTensor) -> SparseTensor:
        DIM = input.coords.shape[-1] - 1
        # upsample scale=2^DIM
        n_cube = torch.ones([2] * DIM, device=input.device, dtype=torch.int)
        n_coords = torch.nonzero(n_cube)
        n_coords = torch.cat([torch.zeros_like(n_coords[:, :1]), n_coords], dim=-1)
        factor = n_coords.shape[0]
        assert factor == 2 ** DIM
        # print(n_coords.shape)
        new_coords = input.coords.clone()
        new_coords[:, 1:] *= 2
        new_coords = new_coords.unsqueeze(1) + n_coords.unsqueeze(0).to(new_coords.dtype)
        
        new_feats = input.feats.unsqueeze(1).expand(input.feats.shape[0], factor, *input.feats.shape[1:])
        out = SparseTensor(new_feats.flatten(0, 1), new_coords.flatten(0, 1), input.shape)
        out._scale = input._scale * 2
        out._spatial_cache = input._spatial_cache
        return out

def sparse_scaled_dot_product_attention(*args, **kwargs):
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
        assert isinstance(qkv, SparseTensor), f"qkv must be a SparseTensor, got {type(qkv)}"
        assert len(qkv.shape) == 4 and qkv.shape[1] == 3, f"Invalid shape for qkv, got {qkv.shape}, expected [N, *, 3, H, C]"
        device = qkv.device

        s = qkv
        q_seqlen = [qkv.layout[i].stop - qkv.layout[i].start for i in range(qkv.shape[0])]
        kv_seqlen = q_seqlen
        qkv = qkv.feats     # [T, 3, H, C]

    elif num_all_args == 2:
        q = args[0] if len(args) > 0 else kwargs['q']
        kv = args[1] if len(args) > 1 else kwargs['kv']
        assert isinstance(q, SparseTensor) and isinstance(kv, (SparseTensor, torch.Tensor)) or \
               isinstance(q, torch.Tensor) and isinstance(kv, SparseTensor), \
               f"Invalid types, got {type(q)} and {type(kv)}"
        assert q.shape[0] == kv.shape[0], f"Batch size mismatch, got {q.shape[0]} and {kv.shape[0]}"
        device = q.device

        if isinstance(q, SparseTensor):
            assert len(q.shape) == 3, f"Invalid shape for q, got {q.shape}, expected [N, *, H, C]"
            s = q
            q_seqlen = [q.layout[i].stop - q.layout[i].start for i in range(q.shape[0])]
            q = q.feats     # [T_Q, H, C]
        else:
            assert len(q.shape) == 4, f"Invalid shape for q, got {q.shape}, expected [N, L, H, C]"
            s = None
            N, L, H, C = q.shape
            q_seqlen = [L] * N
            q = q.reshape(N * L, H, C)   # [T_Q, H, C]

        if isinstance(kv, SparseTensor):
            assert len(kv.shape) == 4 and kv.shape[1] == 2, f"Invalid shape for kv, got {kv.shape}, expected [N, *, 2, H, C]"
            kv_seqlen = [kv.layout[i].stop - kv.layout[i].start for i in range(kv.shape[0])]
            kv = kv.feats     # [T_KV, 2, H, C]
        else:
            assert len(kv.shape) == 5, f"Invalid shape for kv, got {kv.shape}, expected [N, L, 2, H, C]"
            N, L, _, H, C = kv.shape
            kv_seqlen = [L] * N
            kv = kv.reshape(N * L, 2, H, C)   # [T_KV, 2, H, C]

    elif num_all_args == 3:
        q = args[0] if len(args) > 0 else kwargs['q']
        k = args[1] if len(args) > 1 else kwargs['k']
        v = args[2] if len(args) > 2 else kwargs['v']
        assert isinstance(q, SparseTensor) and isinstance(k, (SparseTensor, torch.Tensor)) and type(k) == type(v) or \
               isinstance(q, torch.Tensor) and isinstance(k, SparseTensor) and isinstance(v, SparseTensor), \
               f"Invalid types, got {type(q)}, {type(k)}, and {type(v)}"
        assert q.shape[0] == k.shape[0] == v.shape[0], f"Batch size mismatch, got {q.shape[0]}, {k.shape[0]}, and {v.shape[0]}"
        device = q.device

        if isinstance(q, SparseTensor):
            assert len(q.shape) == 3, f"Invalid shape for q, got {q.shape}, expected [N, *, H, Ci]"
            s = q
            q_seqlen = [q.layout[i].stop - q.layout[i].start for i in range(q.shape[0])]
            q = q.feats     # [T_Q, H, Ci]
        else:
            assert len(q.shape) == 4, f"Invalid shape for q, got {q.shape}, expected [N, L, H, Ci]"
            s = None
            N, L, H, CI = q.shape
            q_seqlen = [L] * N
            q = q.reshape(N * L, H, CI)  # [T_Q, H, Ci]

        if isinstance(k, SparseTensor):
            assert len(k.shape) == 3, f"Invalid shape for k, got {k.shape}, expected [N, *, H, Ci]"
            assert len(v.shape) == 3, f"Invalid shape for v, got {v.shape}, expected [N, *, H, Co]"
            kv_seqlen = [k.layout[i].stop - k.layout[i].start for i in range(k.shape[0])]
            k = k.feats     # [T_KV, H, Ci]
            v = v.feats     # [T_KV, H, Co]
        else:
            assert len(k.shape) == 4, f"Invalid shape for k, got {k.shape}, expected [N, L, H, Ci]"
            assert len(v.shape) == 4, f"Invalid shape for v, got {v.shape}, expected [N, L, H, Co]"
            N, L, H, CI, CO = *k.shape, v.shape[-1]
            kv_seqlen = [L] * N
            k = k.reshape(N * L, H, CI)     # [T_KV, H, Ci]
            v = v.reshape(N * L, H, CO)     # [T_KV, H, Co]


    if ATTN == 'xformers':
        if num_all_args == 1:
            q, k, v = qkv.unbind(dim=1)
        elif num_all_args == 2:
            k, v = kv.unbind(dim=1)
        q = q.unsqueeze(0)
        k = k.unsqueeze(0)
        v = v.unsqueeze(0)
        mask = xops.fmha.BlockDiagonalMask.from_seqlens(q_seqlen, kv_seqlen)
        out = xops.memory_efficient_attention(q, k, v, mask)[0]
    elif ATTN == 'flash_attn':
        cu_seqlens_q = torch.cat([torch.tensor([0]), torch.cumsum(torch.tensor(q_seqlen), dim=0)]).int().to(device)
        if num_all_args in [2, 3]:
            cu_seqlens_kv = torch.cat([torch.tensor([0]), torch.cumsum(torch.tensor(kv_seqlen), dim=0)]).int().to(device)
        if num_all_args == 1:
            out = flash_attn.flash_attn_varlen_qkvpacked_func(qkv, cu_seqlens_q, max(q_seqlen))
        elif num_all_args == 2:
            out = flash_attn.flash_attn_varlen_kvpacked_func(q, kv, cu_seqlens_q, cu_seqlens_kv, max(q_seqlen), max(kv_seqlen))
        elif num_all_args == 3:
            out = flash_attn.flash_attn_varlen_func(q, k, v, cu_seqlens_q, cu_seqlens_kv, max(q_seqlen), max(kv_seqlen))
    else:
        raise ValueError(f"Unknown attention module: {ATTN}")
    
    if s is not None:
        return s.replace(out)
    else:
        return out.reshape(N, L, H, -1)


class SparseMultiHeadRMSNorm(nn.Module):
    def __init__(self, dim: int, heads: int):
        super().__init__()
        self.scale = dim ** 0.5
        self.gamma = nn.Parameter(torch.ones(heads, dim))

    def forward(self, x: Union[SparseTensor, torch.Tensor]) -> Union[SparseTensor, torch.Tensor]:
        x_type = x.dtype
        x = x.float()
        if isinstance(x, SparseTensor):
            x = x.replace(F.normalize(x.feats, dim=-1))
        else:
            x = F.normalize(x, dim=-1)            
        return (x * self.gamma * self.scale).to(x_type)


class SparseMultiHeadAttention(nn.Module):
    def __init__(
        self,
        channels: int,
        num_heads: int,
        ctx_channels: Optional[int] = None,
        type: Literal["self", "cross"] = "self",
        attn_mode: Literal["full", "serialized", "windowed"] = "full",
        window_size: Optional[int] = None,
        shift_sequence: Optional[int] = None,
        shift_window: Optional[Tuple[int, int, int]] = None,
        serialize_mode = None,
        qkv_bias: bool = True,
        use_rope: bool = False,
        qk_rms_norm: bool = False,
    ):
        super().__init__()
        # print(f"== SparseMultiHeadAttention: type={type}, attn_mode={attn_mode}, use_rope={use_rope}, qk_rms_norm={qk_rms_norm}, qkv_bias={qkv_bias}")
        # == SparseMultiHeadAttention: type=self, attn_mode=windowed, use_rope=False, qk_rms_norm=False, qkv_bias=True
        # == SparseMultiHeadAttention: type=self, attn_mode=full, use_rope=False, qk_rms_norm=True, qkv_bias=True
        # == SparseMultiHeadAttention: type=cross, attn_mode=full, use_rope=False, qk_rms_norm=False, qkv_bias=True

        # assert channels == 1024
        # assert num_heads == 16
        # assert ctx_channels == None
        # # assert type == 'self' or 'cross'
        # assert attn_mode == 'full' or 'windowed'
        # assert window_size == None
        assert shift_sequence == None
        # assert shift_window == None
        assert serialize_mode == None
        assert qkv_bias == True
        assert use_rope == False
        # assert qk_rms_norm == True or False

        assert channels % num_heads == 0
        assert type in ["self", "cross"], f"Invalid attention type: {type}"
        assert attn_mode in ["full", "serialized", "windowed"], f"Invalid attention mode: {attn_mode}"
        assert type == "self" or attn_mode == "full", "Cross-attention only supports full attention"
        assert type == "self" or use_rope is False, "Rotary position embeddings only supported for self-attention"
        self.channels = channels
        self.ctx_channels = ctx_channels if ctx_channels is not None else channels
        self.num_heads = num_heads
        self._type = type
        self.attn_mode = attn_mode
        self.window_size = window_size
        # self.shift_sequence = shift_sequence
        self.shift_window = shift_window
        # self.serialize_mode = serialize_mode
        self.use_rope = use_rope
        self.qk_rms_norm = qk_rms_norm

        if self._type == "self":
            self.to_qkv = nn.Linear(channels, channels * 3, bias=qkv_bias)
        else:
            self.to_q = nn.Linear(channels, channels, bias=qkv_bias)
            self.to_kv = nn.Linear(self.ctx_channels, channels * 2, bias=qkv_bias)
        
        if self.qk_rms_norm: # True
            self.q_rms_norm = SparseMultiHeadRMSNorm(channels // num_heads, num_heads)
            self.k_rms_norm = SparseMultiHeadRMSNorm(channels // num_heads, num_heads)
            
        self.to_out = nn.Linear(channels, channels)

        if use_rope: # False
            pdb.set_trace()
            # self.rope = RotaryPositionEmbedder(channels)
        
    @staticmethod
    def _linear(module: nn.Linear, x: Union[SparseTensor, torch.Tensor]) -> Union[SparseTensor, torch.Tensor]:
        if isinstance(x, SparseTensor):
            return x.replace(module(x.feats))
        else:
            return module(x)

    @staticmethod
    def _reshape_chs(x: Union[SparseTensor, torch.Tensor], shape: Tuple[int, ...]) -> Union[SparseTensor, torch.Tensor]:
        if isinstance(x, SparseTensor):
            return x.reshape(*shape)
        else:
            return x.reshape(*x.shape[:2], *shape)

    def _fused_pre(self, x: Union[SparseTensor, torch.Tensor], num_fused: int) -> Union[SparseTensor, torch.Tensor]:
        if isinstance(x, SparseTensor):
            x_feats = x.feats.unsqueeze(0)
        else:
            x_feats = x
        x_feats = x_feats.reshape(*x_feats.shape[:2], num_fused, self.num_heads, -1)
        return x.replace(x_feats.squeeze(0)) if isinstance(x, SparseTensor) else x_feats

    # def _rope(self, qkv: SparseTensor) -> SparseTensor:
    #     q, k, v = qkv.feats.unbind(dim=1)   # [T, H, C]
    #     q, k = self.rope(q, k, qkv.coords[:, 1:])
    #     qkv = qkv.replace(torch.stack([q, k, v], dim=1)) 
    #     return qkv
    
    def forward(self, x: Union[SparseTensor, torch.Tensor], context: Optional[Union[SparseTensor, torch.Tensor]] = None) -> Union[SparseTensor, torch.Tensor]:
        if self._type == "self":
            # == SparseMultiHeadAttention: type=self, attn_mode=windowed, use_rope=False, qk_rms_norm=False, qkv_bias=True
            # == SparseMultiHeadAttention: type=self, attn_mode=full, use_rope=False, qk_rms_norm=True, qkv_bias=True
            qkv = self._linear(self.to_qkv, x)
            qkv = self._fused_pre(qkv, num_fused=3)
            if self.use_rope: # False
                pdb.set_trace()
                # qkv = self._rope(qkv)
            if self.qk_rms_norm: # False | True
                q, k, v = qkv.unbind(dim=1)
                q = self.q_rms_norm(q)
                k = self.k_rms_norm(k)
                qkv = qkv.replace(torch.stack([q.feats, k.feats, v.feats], dim=1))
            if self.attn_mode == "full": # True
                h = sparse_scaled_dot_product_attention(qkv)
            elif self.attn_mode == "serialized":
                pdb.set_trace()
                # h = sparse_serialized_scaled_dot_product_self_attention(
                #     qkv, self.window_size, serialize_mode=self.serialize_mode, shift_sequence=self.shift_sequence, shift_window=self.shift_window
                # )
            elif self.attn_mode == "windowed":
                h = sparse_windowed_scaled_dot_product_self_attention(
                    qkv, self.window_size, shift_window=self.shift_window
                )
        else:
            # == SparseMultiHeadAttention: type=cross, attn_mode=full, use_rope=False, qk_rms_norm=False, qkv_bias=True
            q = self._linear(self.to_q, x)
            q = self._reshape_chs(q, (self.num_heads, -1))
            kv = self._linear(self.to_kv, context)
            kv = self._fused_pre(kv, num_fused=2)
            if self.qk_rms_norm: # True
                pdb.set_trace()
                q = self.q_rms_norm(q)
                k, v = kv.unbind(dim=1)
                k = self.k_rms_norm(k)
                kv = kv.replace(torch.stack([k.feats, v.feats], dim=1))
            h = sparse_scaled_dot_product_attention(q, kv)
        h = self._reshape_chs(h, (-1,))
        h = self._linear(self.to_out, h)
        return h


def calc_window_partition(
    tensor: SparseTensor,
    window_size: Union[int, Tuple[int, ...]],
    shift_window: Union[int, Tuple[int, ...]] = 0
) -> Tuple[torch.Tensor, torch.Tensor, List[int], List[int]]:
    """
    Calculate serialization and partitioning for a set of coordinates.

    Args:
        tensor (SparseTensor): The input tensor.
        window_size (int): The window size to use.
        shift_window (Tuple[int, ...]): The shift of serialized coordinates.

    Returns:
        (torch.Tensor): Forwards indices.
        (torch.Tensor): Backwards indices.
        (List[int]): Sequence lengths.
        (List[int]): Sequence batch indices.
    """
    DIM = tensor.coords.shape[1] - 1
    shift_window = (shift_window,) * DIM if isinstance(shift_window, int) else shift_window
    window_size = (window_size,) * DIM if isinstance(window_size, int) else window_size
    shifted_coords = tensor.coords.clone().detach()
    shifted_coords[:, 1:] += torch.tensor(shift_window, device=tensor.device, dtype=torch.int32).unsqueeze(0)

    MAX_COORDS = shifted_coords[:, 1:].max(dim=0).values.tolist()
    NUM_WINDOWS = [math.ceil((mc + 1) / ws) for mc, ws in zip(MAX_COORDS, window_size)]
    OFFSET = torch.cumprod(torch.tensor([1] + NUM_WINDOWS[::-1]), dim=0).tolist()[::-1]

    shifted_coords[:, 1:] //= torch.tensor(window_size, device=tensor.device, dtype=torch.int32).unsqueeze(0)
    shifted_indices = (shifted_coords * torch.tensor(OFFSET, device=tensor.device, dtype=torch.int32).unsqueeze(0)).sum(dim=1)
    fwd_indices = torch.argsort(shifted_indices)
    bwd_indices = torch.empty_like(fwd_indices)
    bwd_indices[fwd_indices] = torch.arange(fwd_indices.shape[0], device=tensor.device)
    seq_lens = torch.bincount(shifted_indices)
    seq_batch_indices = torch.arange(seq_lens.shape[0], device=tensor.device, dtype=torch.int32) // OFFSET[0]
    mask = seq_lens != 0
    seq_lens = seq_lens[mask].tolist()
    seq_batch_indices = seq_batch_indices[mask].tolist()

    return fwd_indices, bwd_indices, seq_lens, seq_batch_indices
    
def sparse_windowed_scaled_dot_product_self_attention(
    qkv: SparseTensor,
    window_size: int,
    shift_window: Tuple[int, int, int] = (0, 0, 0)
) -> SparseTensor:
    """
    Apply windowed scaled dot product self attention to a sparse tensor.

    Args:
        qkv (SparseTensor): [N, *, 3, H, C] sparse tensor containing Qs, Ks, and Vs.
        window_size (int): The window size to use.
        shift_window (Tuple[int, int, int]): The shift of serialized coordinates.
        shift (int): The shift to use.
    """
    assert len(qkv.shape) == 4 and qkv.shape[1] == 3, f"Invalid shape for qkv, got {qkv.shape}, expected [N, *, 3, H, C]"

    serialization_spatial_cache_name = f'window_partition_{window_size}_{shift_window}'
    serialization_spatial_cache = qkv.get_spatial_cache(serialization_spatial_cache_name)
    if serialization_spatial_cache is None:
        fwd_indices, bwd_indices, seq_lens, seq_batch_indices = calc_window_partition(qkv, window_size, shift_window)
        qkv.register_spatial_cache(serialization_spatial_cache_name, (fwd_indices, bwd_indices, seq_lens, seq_batch_indices))
    else:
        fwd_indices, bwd_indices, seq_lens, seq_batch_indices = serialization_spatial_cache

    M = fwd_indices.shape[0]
    T = qkv.feats.shape[0]
    H = qkv.feats.shape[2]
    C = qkv.feats.shape[3]
    
    qkv_feats = qkv.feats[fwd_indices]      # [M, 3, H, C]


    if all([seq_len == window_size for seq_len in seq_lens]):
        B = len(seq_lens)
        N = window_size
        qkv_feats = qkv_feats.reshape(B, N, 3, H, C)
        if ATTN == 'xformers':
            q, k, v = qkv_feats.unbind(dim=2)                       # [B, N, H, C]
            out = xops.memory_efficient_attention(q, k, v)          # [B, N, H, C]
        elif ATTN == 'flash_attn':
            out = flash_attn.flash_attn_qkvpacked_func(qkv_feats)   # [B, N, H, C]
        else:
            raise ValueError(f"Unknown attention module: {ATTN}")
        out = out.reshape(B * N, H, C)                              # [M, H, C]
    else:
        if ATTN == 'xformers':
            q, k, v = qkv_feats.unbind(dim=1)                       # [M, H, C]
            q = q.unsqueeze(0)                                      # [1, M, H, C]
            k = k.unsqueeze(0)                                      # [1, M, H, C]
            v = v.unsqueeze(0)                                      # [1, M, H, C]
            mask = xops.fmha.BlockDiagonalMask.from_seqlens(seq_lens)
            out = xops.memory_efficient_attention(q, k, v, mask)[0] # [M, H, C]
        elif ATTN == 'flash_attn':
            cu_seqlens = torch.cat([torch.tensor([0]), torch.cumsum(torch.tensor(seq_lens), dim=0)], dim=0) \
                        .to(qkv.device).int()
            out = flash_attn.flash_attn_varlen_qkvpacked_func(qkv_feats, cu_seqlens, max(seq_lens)) # [M, H, C]

    out = out[bwd_indices]      # [T, H, C]



    return qkv.replace(out)


class SparseConv3d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, dilation=1, padding=None, bias=True, indice_key=None):
        super(SparseConv3d, self).__init__()
        # in_channels = 768
        # out_channels = 192
        # kernel_size = 3
        # stride = 1
        # dilation = 1
        # padding = None
        # bias = True
        # indice_key = 'res_128'
        if 'spconv' not in globals(): # True
            import spconv.pytorch as spconv

        # assert SPCONV_ALGO == 'native'
        algo = spconv.ConvAlgo.Native
        if stride == 1 and (padding is None): # True
            self.conv = spconv.SubMConv3d(in_channels, out_channels, kernel_size, dilation=dilation, bias=bias, indice_key=indice_key, algo=algo)
        else:
            self.conv = spconv.SparseConv3d(in_channels, out_channels, kernel_size, stride=stride, dilation=dilation, padding=padding, bias=bias, indice_key=indice_key, algo=algo)
        self.stride = tuple(stride) if isinstance(stride, (list, tuple)) else (stride, stride, stride)
        # self.stride --- (1, 1, 1)
        self.padding = padding # None
        # self.conv -- SubMConv3d(768, 192, kernel_size=[3, 3, 3], stride=[1, 1, 1], padding=[0, 0, 0], dilation=[1, 1, 1], output_padding=[0, 0, 0], algo=ConvAlgo.Native)
        # assert self.padding == None
        
    def forward(self, x: SparseTensor) -> SparseTensor:
        spatial_changed = any(s != 1 for s in self.stride) or (self.padding is not None)
        new_data = self.conv(x.data)

        new_shape = [x.shape[0], self.conv.out_channels]
        new_layout = None if spatial_changed else x.layout

        if spatial_changed and (x.shape[0] != 1):
            # spconv was non-1 stride will break the contiguous of the output tensor, sort by the coords
            fwd = new_data.indices[:, 0].argsort()
            bwd = torch.zeros_like(fwd).scatter_(0, fwd, torch.arange(fwd.shape[0], device=fwd.device))
            sorted_feats = new_data.features[fwd]
            sorted_coords = new_data.indices[fwd]
            unsorted_data = new_data
            new_data = spconv.SparseConvTensor(sorted_feats, sorted_coords, unsorted_data.spatial_shape, unsorted_data.batch_size)  # type: ignore

        out = SparseTensor(
            new_data, shape=torch.Size(new_shape), layout=new_layout,
            scale=tuple([s * stride for s, stride in zip(x._scale, self.stride)]),
            spatial_cache=x._spatial_cache,
        )

        if spatial_changed and (x.shape[0] != 1):
            out.register_spatial_cache(f'conv_{self.stride}_unsorted_data', unsorted_data)
            out.register_spatial_cache(f'conv_{self.stride}_sort_bwd', bwd)
 
        return out


class SparseFeedForwardNet(nn.Module):
    def __init__(self, channels: int, mlp_ratio: float = 4.0):
        super().__init__()
        self.mlp = nn.Sequential(
            SparseLinear(channels, int(channels * mlp_ratio)),
            SparseGELU(approximate="tanh"),
            SparseLinear(int(channels * mlp_ratio), channels),
        )
        # channels = 1024
        # mlp_ratio = 4

    def forward(self, x: SparseTensor) -> SparseTensor:
        return self.mlp(x)


