""""""
Byte packing utilities for canonical hash output
Converts numeric arrays to deterministic byte representations
""""""

import numpy as np

def to_packed_bytes(data):
    """"""
    Canonical packing: zscore normalization -> clip to ±6sigma -> uint32
    Ensures deterministic, high-entropy byte representation
    """"""
    if isinstance(data, (list, tuple)):
        data = np.array(data, dtype=complex)

    # Handle complex data by taking magnitude and phase
    if np.iscomplexobj(data):
        # Interleave real and imaginary parts
        real_part = np.real(data)
        imag_part = np.imag(data)
        combined = np.empty(real_part.size + imag_part.size, dtype=real_part.dtype)
        combined[0::2] = real_part
        combined[1::2] = imag_part
        data = combined

    # Ensure we have floating point data
    data = np.asarray(data, dtype=np.float64)

    # Z-score normalization
    mean_val = np.mean(data)
    std_val = np.std(data)
    if std_val > 0:
        normalized = (data - mean_val) / std_val
    else:
        normalized = data - mean_val

    # Clip to ±6sigma to prevent outliers from dominating
    clipped = np.clip(normalized, -6.0, 6.0)

    # Scale to uint32 range
    # Map [-6, 6] to [0, 2^32-1]
    scaled = ((clipped + 6.0) / 12.0) * (2**32 - 1)
    uint32_data = scaled.astype(np.uint32)

    return uint32_data.tobytes()

def squeeze_to_hash_size(data, target_bytes=32):
    """"""
    Squeeze byte array to target size using XOR folding
    Preserves entropy while reducing length
    """"""
    if len(data) <= target_bytes:
        # Pad with zeros if too short
        return data + b'\x00' * (target_bytes - len(data))

    # XOR-fold longer data into target size
    result = bytearray(target_bytes)
    for i, byte_val in enumerate(data):
        result[i % target_bytes] ^= byte_val

    return bytes(result)
