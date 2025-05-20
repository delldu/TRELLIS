from typing import *
import torch
from .. import SparseTensor
import xformers.ops as xops
import todos
import pdb

__all__ = [
    'sparse_scaled_dot_product_attention',
]


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
        # (qkv)
        qkv = args[0] if len(args) > 0 else kwargs['qkv']
        assert isinstance(qkv, SparseTensor), f"qkv must be a SparseTensor, got {type(qkv)}"
        assert len(qkv.shape) == 4 and qkv.shape[1] == 3, f"Invalid shape for qkv, got {qkv.shape}, expected [N, *, 3, H, C]"
        device = qkv.device

        s = qkv
        q_seqlen = [qkv.layout[i].stop - qkv.layout[i].start for i in range(qkv.shape[0])] # xxxx_3333
        kv_seqlen = q_seqlen
        qkv = qkv.feats     # [T, 3, H, C]

    elif num_all_args == 2:
        # (q, kv)
        q = args[0] if len(args) > 0 else kwargs['q']
        kv = args[1] if len(args) > 1 else kwargs['kv']
        assert isinstance(q, SparseTensor) and isinstance(kv, (SparseTensor, torch.Tensor)) or \
               isinstance(q, torch.Tensor) and isinstance(kv, SparseTensor), \
               f"Invalid types, got {type(q)} and {type(kv)}"
        assert q.shape[0] == kv.shape[0], f"Batch size mismatch, got {q.shape[0]} and {kv.shape[0]}"
        device = q.device

        assert isinstance(q, SparseTensor) == True
        assert len(q.shape) == 3, f"Invalid shape for q, got {q.shape}, expected [N, *, H, C]"
        s = q
        q_seqlen = [q.layout[i].stop - q.layout[i].start for i in range(q.shape[0])]
        q = q.feats     # [T_Q, H, C]

        assert len(kv.shape) == 5, f"Invalid shape for kv, got {kv.shape}, expected [N, L, 2, H, C]"
        N, L, _, H, C = kv.shape
        kv_seqlen = [L] * N
        kv = kv.reshape(N * L, 2, H, C)   # [T_KV, 2, H, C]

    elif num_all_args == 3:
        pdb.set_trace()

        q = args[0] if len(args) > 0 else kwargs['q']
        k = args[1] if len(args) > 1 else kwargs['k']
        v = args[2] if len(args) > 2 else kwargs['v']
        assert isinstance(q, SparseTensor) and isinstance(k, (SparseTensor, torch.Tensor)) and type(k) == type(v) or \
               isinstance(q, torch.Tensor) and isinstance(k, SparseTensor) and isinstance(v, SparseTensor), \
               f"Invalid types, got {type(q)}, {type(k)}, and {type(v)}"
        assert q.shape[0] == k.shape[0] == v.shape[0], f"Batch size mismatch, got {q.shape[0]}, {k.shape[0]}, and {v.shape[0]}"
        device = q.device

        assert isinstance(q, SparseTensor) == True
        assert len(q.shape) == 3, f"Invalid shape for q, got {q.shape}, expected [N, *, H, Ci]"
        s = q
        q_seqlen = [q.layout[i].stop - q.layout[i].start for i in range(q.shape[0])]
        q = q.feats     # [T_Q, H, Ci]

        assert isinstance(k, SparseTensor) == True

        assert len(k.shape) == 3, f"Invalid shape for k, got {k.shape}, expected [N, *, H, Ci]"
        assert len(v.shape) == 3, f"Invalid shape for v, got {v.shape}, expected [N, *, H, Co]"
        kv_seqlen = [k.layout[i].stop - k.layout[i].start for i in range(k.shape[0])]
        k = k.feats     # [T_KV, H, Ci]
        v = v.feats     # [T_KV, H, Co]

    if num_all_args == 1:
        q, k, v = qkv.unbind(dim=1)
    elif num_all_args == 2:
        k, v = kv.unbind(dim=1)
    q = q.unsqueeze(0)
    k = k.unsqueeze(0)
    v = v.unsqueeze(0)

    #pdb.set_trace()
    mask = xops.fmha.BlockDiagonalMask.from_seqlens(q_seqlen, kv_seqlen)
    out = xops.memory_efficient_attention(q, k, v, mask)[0]
    # tensor [out] size: [3185, 16, 64], min: -0.679199, max: 0.706543, mean: 0.001805
    
    return s.replace(out, s.coords)
