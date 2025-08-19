# encryption/finalizer.py
import numpy as np

MASK32 = np.uint64(0xFFFFFFFF)
UINT32 = np.uint32

def mix32(x: np.uint32) -> np.uint32:
    """Jenkins-like avalanching mix with safe 64-bit arithmetic."""
    z = np.uint64(x) & MASK32
    z ^= (z >> np.uint64(15)); z = (z * np.uint64(0x2C1B3C6D)) & MASK32
    z ^= (z >> np.uint64(12)); z = (z * np.uint64(0x297A2D39)) & MASK32
    z ^= (z >> np.uint64(15))
    return UINT32(z)

def finalize_words(words: np.ndarray) -> np.ndarray:
    """Apply position-dependent finalizing mix to each word."""
    if words.dtype != np.uint32:
        words = words.astype(np.uint32)

    result = np.zeros_like(words)
    for i in range(len(words)):
        s = np.uint32((i * 0x9E3779B1) & 0xFFFFFFFF)
        result[i] = mix32(UINT32((words[i] ^ s) & 0xFFFFFFFF))

    return result
