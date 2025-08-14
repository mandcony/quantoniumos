# encryption/mds.py
import numpy as np

def _rotl32(x, r): 
    return ((x << r) & 0xFFFFFFFF) | (x >> (32 - r))

def mds_mix32(words: np.ndarray) -> np.ndarray:
    """
    In-place 4x4 word MDS-like linear layer over Z/2^32Z using rotations/XOR.
    Branch-friendly and fast. Length must be multiple of 4.
    """
    assert words.dtype == np.uint32
    n = len(words)
    assert n % 4 == 0
    for i in range(0, n, 4):
        a,b,c,d = [int(words[i+j]) for j in range(4)]
        x = (a ^ _rotl32(b, 1)  ^ _rotl32(c, 2)  ^ _rotl32(d, 7))  & 0xFFFFFFFF
        y = (_rotl32(a, 3)  ^ b ^ _rotl32(c, 5)  ^ _rotl32(d, 9))  & 0xFFFFFFFF
        z = (_rotl32(a, 13) ^ _rotl32(b, 7)  ^ c ^ _rotl32(d, 11)) & 0xFFFFFFFF
        w = (_rotl32(a, 17) ^ _rotl32(b, 19) ^ _rotl32(c, 23) ^ d) & 0xFFFFFFFF
        words[i:i+4] = (x, y, z, w)
    return words

def keyed_weyl_add(words: np.ndarray, key_words: np.ndarray, ctr: int) -> None:
    """
    Cheap keyed 'round constants' via a Weyl progression, different per word index.
    Keeps distribution flatter across inputs.
    """
    C = 0x9E3779B1  # φ*2^32
    s = (ctr * C) & 0xFFFFFFFF
    for j in range(len(words)):
        k = key_words[j % len(key_words)]
        rc = (s ^ _rotl32(s, (j & 31))) & 0xFFFFFFFF
        words[j] = (words[j] + k + rc) & 0xFFFFFFFF
        s = (s + C) & 0xFFFFFFFF
