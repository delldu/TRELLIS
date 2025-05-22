from typing import *
import torch
from .. import SparseTensor
import xformers.ops as xops
import todos
import pdb

__all__ = [
    'sparse_scaled_dot_product_attention',
]

def sparse_scaled_dot_product_attention(*args):
    arg_names_dict = {
        1: ['qkv'],
        2: ['q', 'kv'],
        3: ['q', 'k', 'v']
    }
    num_all_args = len(args)

    if num_all_args == 1: # (qkv)
        qkv = args[0]
        assert isinstance(qkv, SparseTensor)
        assert len(qkv.shape) == 4 and qkv.shape[1] == 3
        device = qkv.device

        s = qkv
        q_seqlen = [qkv.coords.size(0) for i in range(qkv.shape[0])] # xxxx_3333
        kv_seqlen = q_seqlen
        qkv = qkv.feats     # [T, 3, H, C]
        q, k, v = qkv.unbind(dim=1)

    elif num_all_args == 2: # (q, kv)
        q = args[0]
        kv = args[1]
        assert isinstance(q, SparseTensor)
        assert isinstance(kv, (SparseTensor, torch.Tensor))
        assert q.shape[0] == kv.shape[0]
        device = q.device

        assert len(q.shape) == 3
        s = q
        q_seqlen = [q.coords.size(0) for i in range(q.shape[0])] # xxxx_3333
        q = q.feats     # [T_Q, H, C]

        assert len(kv.shape) == 5
        N, L, _, H, C = kv.shape
        kv_seqlen = [L] * N
        kv = kv.reshape(N * L, 2, H, C)   # [T_KV, 2, H, C]
        k, v = kv.unbind(dim=1)

    q = q.unsqueeze(0)
    k = k.unsqueeze(0)
    v = v.unsqueeze(0)

    mask = xops.fmha.BlockDiagonalMask.from_seqlens(q_seqlen, kv_seqlen)
    out = xops.memory_efficient_attention(q, k, v, mask)[0]
    # tensor [out] size: [3185, 16, 64], min: -0.679199, max: 0.706543, mean: 0.001805
    
    return s.replace(out, s.coords)
