"""
Wide-pipe diffusion for ultra-low avalanche variance.
Increases state size to 16 words (512 bits) for better uniformity.
"""

import numpy as np
from .mds import mds_mix32, keyed_weyl_add
from .finalizer import finalize_words

MASK32 = np.uint64(0xFFFFFFFF)
UINT32 = np.uint32

def rotl32(x, r: int) -> np.uint32:
    # rotation in 64-bit lane, masked to 32 on exit
    x64 = np.uint64(x) & MASK32
    r = int(r) % 32
    return UINT32(((x64 << r) | (x64 >> (32 - r))) & MASK32)

def add32(*vals) -> np.uint32:
    s = np.uint64(0)
    for v in vals:
        s = (s + (np.uint64(v) & MASK32)) & MASK32
    return UINT32(s)

def xor32(a, b) -> np.uint32:
    return UINT32((np.uint64(a) ^ np.uint64(b)) & MASK32)

def ensure_u32(arr: np.ndarray) -> np.ndarray:
    # in-place view when possible
    if arr.dtype != np.uint32:
        arr = arr.astype(np.uint32, copy=False)
    return arr

def load_round_keys(key_bytes: bytes, words_per_round: int) -> np.ndarray:
    # 64B key schedule base, repeat/trim to words_per_round
    base = np.frombuffer((key_bytes + b"\x00"*64)[:64], dtype=np.uint32)
    if base.size < words_per_round:
        reps = (words_per_round + base.size - 1) // base.size
        base = np.tile(base, reps)
    return base[:words_per_round].astype(np.uint32, copy=False)


def wide_arx_round(words: np.ndarray, round_keys: np.ndarray) -> np.ndarray:
    """
    words: np.uint32 length multiple of 4
    round_keys: np.uint32 array (>= len(words)//4)
    """
    w = ensure_u32(words)
    rk = ensure_u32(round_keys)
    for i in range(0, w.size, 4):
        a = UINT32(w[i+0]); b = UINT32(w[i+1]); c = UINT32(w[i+2]); d = UINT32(w[i+3])
        k = UINT32(rk[(i // 4) % rk.size])

        # ARX quarter-round with safe adds
        a = add32(a, b, k); d = xor32(d, a); d = rotl32(d, 16)
        c = add32(c, d);    b = xor32(b, c); b = rotl32(b, 12)
        a = add32(a, b);    d = xor32(d, a); d = rotl32(d, 8)
        c = add32(c, d);    b = xor32(b, c); b = rotl32(b, 7)

        w[i+0], w[i+1], w[i+2], w[i+3] = a, b, c, d
    return w


def wide_keyed_diffusion(words_or_bytes, key: bytes, rounds: int = 3) -> bytes:
    """
    Accepts bytes or uint32 array; returns bytes.
    Uses wide_arx_round in mod-2^32 with safe arithmetic.
    """
    if isinstance(words_or_bytes, (bytes, bytearray, memoryview)):
        pad = (64 - (len(words_or_bytes) % 64)) % 64  # 16 words
        buf = bytes(words_or_bytes) + b"\x00" * pad
        w = np.frombuffer(buf, dtype=np.uint32).copy()
    else:
        w = np.array(words_or_bytes, dtype=np.uint32, copy=True)
        padw = (-w.size) % 16
        if padw:
            w = np.concatenate([w, np.zeros(padw, dtype=np.uint32)])

    rk = load_round_keys(key, words_per_round=max(4, w.size // 4))

    for r in range(rounds):
        # simple per-round tweak to keys (Weyl progression in u32)
        C = np.uint32(0x9E3779B1)
        tweak = (np.uint32(r) * C)  # u32
        rk_round = (rk + tweak).astype(np.uint32, copy=False)
        w = wide_arx_round(w, rk_round)

    return (w.astype(np.uint32, copy=False)).tobytes()
