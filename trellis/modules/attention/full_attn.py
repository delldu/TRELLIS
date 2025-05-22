from typing import *
import torch
import math
import xformers.ops as xops
import pdb

__all__ = [
    'scaled_dot_product_attention',
]

def scaled_dot_product_attention(*args, **kwargs):
    arg_names_dict = {
        1: ['qkv'],
        2: ['q', 'kv'],
        3: ['q', 'k', 'v']
    }

    num_all_args = len(args) + len(kwargs)
    assert len(kwargs) == 0

    # assert num_all_args in arg_names_dict, f"Invalid number of arguments, got {num_all_args}, expected 1, 2, or 3"
    # for key in arg_names_dict[num_all_args][len(args):]:
    #     assert key in kwargs, f"Missing argument {key}"

    if num_all_args == 1:
        pdb.set_trace()

        # qkv = args[0] if len(args) > 0 else kwargs['qkv']
        # assert len(qkv.shape) == 5 and qkv.shape[2] == 3, f"Invalid shape for qkv, got {qkv.shape}, expected [N, L, 3, H, C]"
        # device = qkv.device
    elif num_all_args == 2: # True | False
        q = args[0]
        kv = args[1]
        assert q.shape[0] == kv.shape[0]
        assert len(q.shape) == 4
        assert len(kv.shape) == 5
        device = q.device
    elif num_all_args == 3: # True | False
        q = args[0]
        k = args[1]
        v = args[2]
        assert q.shape[0] == k.shape[0] == v.shape[0]
        assert len(q.shape) == 4
        assert len(k.shape) == 4
        assert len(v.shape) == 4
        device = q.device    

    if num_all_args == 1:
        pdb.set_trace()
        # q, k, v = qkv.unbind(dim=2)
    elif num_all_args == 2:
        k, v = kv.unbind(dim=2)
    out = xops.memory_efficient_attention(q, k, v)        
    
    return out
