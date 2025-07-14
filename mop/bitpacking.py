# mop/bit_packing.py
"""
Row-group-local bit-packing helpers
----------------------------------
• pack(ids)   -> (bytes, bit_width)
• unpack(buf, bit_width, n) -> np.ndarray
"""

from __future__ import annotations
import math
import numpy as np
from bitarray import bitarray
from bitarray.util import int2ba, ba2int


def pack(ids: np.ndarray) -> tuple[bytes, int]:
    """Return (raw_byte_stream, bit_width)."""
    assert ids.ndim == 1
    bw = max(1, int(ids.max()).bit_length())
    bits = bitarray(endian="little")
    for x in ids:
        bits += int2ba(int(x), length=bw, endian="little")
    return bits.tobytes(), bw


def unpack(buf: bytes, bw: int, n: int) -> np.ndarray:
    """Inverse of `pack`."""
    bits = bitarray(endian="little")
    bits.frombytes(buf)
    vals = [ba2int(bits[i * bw : (i + 1) * bw], endian="little") for i in range(n)]
    return np.asarray(vals, dtype=np.uint32)
