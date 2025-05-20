from typing import *
import torch
import math
from .. import SparseTensor
from .. import ATTN
import todos
import pdb

if ATTN == 'xformers':
    import xformers.ops as xops
elif ATTN == 'flash_attn':
    import flash_attn
else:
    raise ValueError(f"Unknown attention module: {ATTN}")


__all__ = [
    'sparse_windowed_scaled_dot_product_self_attention',
]


def calc_window_partition(
    tensor: SparseTensor,
    window_size: Union[int, Tuple[int, ...]],
    shift_window: Union[int, Tuple[int, ...]] = 0
) -> Tuple[torch.Tensor, torch.Tensor, List[int], List[int]]:
    """
    Calculate serialization and partitioning for a set of coordinates.
    """
    # tensor.shape = torch.Size([1, 3, 12, 64]), 

    # tensor.coords.shape -- [14955, 4]
    DIM = tensor.coords.shape[1] - 1
    assert DIM == 3
    assert isinstance(shift_window, int) == True
    assert isinstance(window_size, int) == True
    assert window_size == 8

    shift_window = (shift_window,) * DIM if isinstance(shift_window, int) else shift_window
    window_size = (window_size,) * DIM if isinstance(window_size, int) else window_size
    shifted_coords = tensor.coords.clone().detach()
    # shifted_coords.size() -- [14955, 4]
    # torch.tensor(shift_window, device=tensor.device, dtype=torch.int32).unsqueeze(0) == 0 or 4 ...
    shifted_coords[:, 1:] += torch.tensor(shift_window, device=tensor.device, dtype=torch.int32).unsqueeze(0)

    MAX_COORDS = shifted_coords[:, 1:].max(dim=0).values.tolist() # [59, 45, 63]
    NUM_WINDOWS = [math.ceil((mc + 1) / ws) for mc, ws in zip(MAX_COORDS, window_size)] # NUM_WINDOWS -- [8, 6, 8]
    OFFSET = torch.cumprod(torch.tensor([1] + NUM_WINDOWS[::-1]), dim=0).tolist()[::-1]
    # OFFSET -- [384, 48, 8, 1]

    shifted_coords[:, 1:] //= torch.tensor(window_size, device=tensor.device, dtype=torch.int32).unsqueeze(0)
    # shifted_coords[:, 1:] //= window_size
    shifted_indices = (shifted_coords * torch.tensor(OFFSET, device=tensor.device, dtype=torch.int32).unsqueeze(0)).sum(dim=1)

    fwd_indices = torch.argsort(shifted_indices) # fwd_indices.size() -- [14955]
    bwd_indices = torch.empty_like(fwd_indices)
    bwd_indices[fwd_indices] = torch.arange(fwd_indices.shape[0], device=tensor.device) # fwd_indices.shape[0] --- 14955


    seq_lens = torch.bincount(shifted_indices)
    # seq_lens.size() -- torch.Size([384]), [0, 0, 0,  ..., 0, 6, 2]
    # seq_batch_indices = torch.arange(seq_lens.shape[0], device=tensor.device, dtype=torch.int32) // OFFSET[0]
    mask = seq_lens != 0
    seq_lens = seq_lens[mask].tolist()
    # seq_batch_indices = seq_batch_indices[mask].tolist()

    # return fwd_indices, bwd_indices, seq_lens, seq_batch_indices
    # tensor [fwd_indices] size: [14955], min: 0.0, max: 14954.0, mean: 7476.999023
    # tensor [bwd_indices] size: [14955], min: 0.0, max: 14954.0, mean: 7476.999512
    # seq_lens is list: len = 134
    return fwd_indices, bwd_indices, seq_lens

    
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

    # qkv.shape = torch.Size([1, 3, 12, 64]), 
    assert len(qkv.shape) == 4 and qkv.shape[1] == 3, f"Invalid shape for qkv, got {qkv.shape}, expected [N, 3, H, C]"
    assert window_size == 8
    assert shift_window == 0 or shift_window == 4

    fwd_indices, bwd_indices, seq_lens = calc_window_partition(qkv, window_size, shift_window)

    # M = fwd_indices.shape[0] #  tensor [fwd_indices] size: [14955], min: 0.0, max: 14954.0, mean: 7476.999023
    T = qkv.feats.shape[0]
    H = qkv.feats.shape[2]
    C = qkv.feats.shape[3]
    qkv_feats = qkv.feats[fwd_indices]      # [M, 3, H, C]

    # print(f"qkv.shape = {qkv.shape}, seq_lens = {seq_lens} ...")
    # qkv.shape = torch.Size([1, 3, 12, 64]), 
    # seq_lens = [6, 2, 18, 4, 21, 46, 56, 4, 18, 2, 35, 50, 95, 273, 395, 140, 
    #     9, 12, 38, 159, 60, 2, 193, 143, 1, 20, 61, 4, 147, 310, 11, 1, 
    #     9, 55, 194, 30, 173, 198, 15, 26, 85, 6, 9, 14, 213, 94, 40, 85, 
    #     91, 72, 48, 56, 4, 200, 286, 217, 192, 224, 267, 217, 183, 18, 156, 149, 
    #     57, 37, 24, 20, 34, 64, 6, 2, 3, 129, 32, 36, 39, 58, 34, 62, 66, 6, 
    #     179, 166, 171, 147, 184, 157, 186, 202, 30, 96, 24, 39, 33, 44, 39, 57, 73, 
    #     6, 6, 1, 164, 101, 25, 39, 35, 53, 65, 67, 6, 200, 305, 236, 238, 234, 216, 
    #     238, 259, 24, 147, 110, 37, 80, 34, 82, 80, 84, 9, 8, 2, 204, 116, 2, 26, 56, 4, 
    #     158, 217, 5, 6, 102, 260, 86, 186, 139, 46, 85, 17, 9, 34, 1, 4, 30, 92, 21, 27, 13, 58, 
    #     210, 386, 141, 36, 8, 17, 57, 136, 217, 52, 2]


    # tensor [qkv_feats] size: [14955, 3, 12, 64], min: -18.796875, max: 18.203125, mean: -0.034009
    q, k, v = qkv_feats.unbind(dim=1)                       # [M, H, C]
    q = q.unsqueeze(0)                                      # [1, M, H, C]
    k = k.unsqueeze(0)                                      # [1, M, H, C]
    v = v.unsqueeze(0)                                      # [1, M, H, C]
    mask = xops.fmha.BlockDiagonalMask.from_seqlens(seq_lens)
    out = xops.memory_efficient_attention(q, k, v, mask)[0] # [M, H, C]

    # tensor [out] size: [14955, 12, 64], min: -16.890625, max: 8.914062, mean: -0.022558
    out = out[bwd_indices]      # [T, H, C]
    # tensor [out] size: [14955, 12, 64], min: -16.890625, max: 8.914062, mean: -0.022558

    return qkv.replace(out)
