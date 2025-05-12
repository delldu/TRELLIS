from typing import *
from enum import Enum

from .. import DEBUG, ATTN

if ATTN == 'xformers':
    import xformers.ops as xops
elif ATTN == 'flash_attn':
    import flash_attn
else:
    raise ValueError(f"Unknown attention module: {ATTN}")


__all__ = [
    # 'sparse_serialized_scaled_dot_product_self_attention',
]


class SerializeMode(Enum):
    Z_ORDER = 0
    Z_ORDER_TRANSPOSED = 1
    HILBERT = 2
    HILBERT_TRANSPOSED = 3


SerializeModes = [
    SerializeMode.Z_ORDER,
    SerializeMode.Z_ORDER_TRANSPOSED,
    SerializeMode.HILBERT,
    SerializeMode.HILBERT_TRANSPOSED
]

