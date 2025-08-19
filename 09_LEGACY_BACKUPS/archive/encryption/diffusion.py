import numpy as np
from encryption.mds import mds_mix32, keyed_weyl_add, _rotl32

def _qr(a, b, c, d):
    a = np.uint32((np.uint64(a) + np.uint64(b)) & 0xFFFFFFFF); d = np.uint32(d) ^ a; d = _rotl32(d, 16)
    c = np.uint32((np.uint64(c) + np.uint64(d)) & 0xFFFFFFFF); b = np.uint32(b) ^ c; b = _rotl32(b, 12)
    a = np.uint32((np.uint64(a) + np.uint64(b)) & 0xFFFFFFFF); d = np.uint32(d) ^ a; d = _rotl32(d, 8)
    c = np.uint32((np.uint64(c) + np.uint64(d)) & 0xFFFFFFFF); b = np.uint32(b) ^ c; b = _rotl32(b, 7)
    return a, b, c, d

def _inv_qr(a, b, c, d):
    b = (b ^ c); b = ((b >> 7) | ((b & 0x7F) << 25)) & 0xFFFFFFFF
    c = (c - d) & 0xFFFFFFFF
    d = (d ^ ( (a + b) & 0xFFFFFFFF )); d = ((d >> 8) | ((d & 0xFF) << 24)) & 0xFFFFFFFF
    a = (a - b) & 0xFFFFFFFF
    b = (b ^ c); b = ((b >> 12) | ((b & 0xFFF) << 20)) & 0xFFFFFFFF
    c = (c - d) & 0xFFFFFFFF
    d = (d ^ ( (a + b) & 0xFFFFFFFF )); d = ((d >> 16) | ((d & 0xFFFF) << 16)) & 0xFFFFFFFF
    a = (a - b) & 0xFFFFFFFF
    return a, b, c, d

def _words_from_bytes(b):
    pad = (-len(b)) % 16
    if pad:
        b = b + bytes(pad)
    arr = np.frombuffer(b, dtype=np.uint32, count=len(b)//4)
    return arr.copy()

def _bytes_from_words(w):
    return (w.astype(np.uint32)).tobytes()

def _subkeys(key: bytes, rounds: int):
    seed = np.frombuffer((key + b"\x00"*64)[:64], dtype=np.uint32)
    ks = []
    for r in range(rounds):
        a = seed.copy()
        for i in range(0, 16, 4):
            a[i:i+4] = _qr(*a[i:i+4])
        ks.append(a.copy())
        seed = np.roll(a, 3)
    return ks

def keyed_nonlinear_diffusion(data: bytes, key: bytes, rounds: int = 2) -> bytes:
    w = _words_from_bytes(data)
    padw = (-len(w)) % 16
    if padw:
        w = np.concatenate([w, np.zeros(padw, dtype=np.uint32)])
    ks = _subkeys(key, rounds)
    for blk in range(0, len(w), 16):
        v = w[blk:blk+16].copy()
        for r in range(rounds):
            k = ks[r]
            v = (v + k) & 0xFFFFFFFF
            v[0], v[4], v[8],  v[12] = _qr(v[0], v[4], v[8],  v[12])
            v[1], v[5], v[9],  v[13] = _qr(v[1], v[5], v[9],  v[13])
            v[2], v[6], v[10], v[14] = _qr(v[2], v[6], v[10], v[14])
            v[3], v[7], v[11], v[15] = _qr(v[3], v[7], v[11], v[15])
            v[0], v[5], v[10], v[15] = _qr(v[0], v[5], v[10], v[15])
            v[1], v[6], v[11], v[12] = _qr(v[1], v[6], v[11], v[12])
            v[2], v[7], v[8],  v[13] = _qr(v[2], v[7], v[8],  v[13])
            v[3], v[4], v[9],  v[14] = _qr(v[3], v[4], v[9],  v[14])
        w[blk:blk+16] = v
    return _bytes_from_words(w)

def keyed_nonlinear_diffusion_v2(data: bytes, key: bytes, rounds: int = 3) -> bytes:
    """
    Stronger keyed diffusion: Weyl adds -> MDS mixing -> ARX.
    Deterministic; same IO contract as v1.
    """
    # bytes -> uint32 words (LE), pad to 16-word blocks
    w = np.frombuffer(data + b"\x00" * ((64 - (len(data) % 64)) % 64), dtype=np.uint32).copy()
    key_words = np.frombuffer((key + b"\x00"*64)[:64], dtype=np.uint32).copy()

    def qr(a, b, c, d):
        a = (a + b) & 0xFFFFFFFF; d ^= a; d = _rotl32(d, 16)
        c = (c + d) & 0xFFFFFFFF; b ^= c; b = _rotl32(b, 12)
        a = (a + b) & 0xFFFFFFFF; d ^= a; d = _rotl32(d, 8)
        c = (c + d) & 0xFFFFFFFF; b ^= c; b = _rotl32(b, 7)
        return a, b, c, d

    for blk in range(0, len(w), 16):
        v = w[blk:blk+16].copy()
        for r in range(rounds):
            # 1) keyed adds (Weyl-derived)
            keyed_weyl_add(v, key_words, ctr=r + blk//16)

            # 2) MDS mix (uniform linear spreading)
            mds_mix32(v)

            # 3) ARX (nonlinear)
            v[0], v[4], v[8],  v[12] = qr(v[0], v[4], v[8],  v[12])
            v[1], v[5], v[9],  v[13] = qr(v[1], v[5], v[9],  v[13])
            v[2], v[6], v[10], v[14] = qr(v[2], v[6], v[10], v[14])
            v[3], v[7], v[11], v[15] = qr(v[3], v[7], v[11], v[15])
            v[0], v[5], v[10], v[15] = qr(v[0], v[5], v[10], v[15])
            v[1], v[6], v[11], v[12] = qr(v[1], v[6], v[11], v[12])
            v[2], v[7], v[8],  v[13] = qr(v[2], v[7], v[8],  v[13])
            v[3], v[4], v[9],  v[14] = qr(v[3], v[4], v[9],  v[14])

        w[blk:blk+16] = v
    return (w.astype(np.uint32)).tobytes()
