"""
Direct test of improved hash without problematic imports
"""

import numpy as np
import sys
import os
sys.path.append('.')

from encryption.diffusion import keyed_nonlinear_diffusion

def to_packed_bytes(data):
    """Canonical packing: zscore normalization -> clip to ±6σ -> uint32"""
    if isinstance(data, (list, tuple)):
        data = np.array(data, dtype=complex)
    
    if np.iscomplexobj(data):
        real_part = np.real(data)
        imag_part = np.imag(data)
        combined = np.empty(real_part.size + imag_part.size, dtype=real_part.dtype)
        combined[0::2] = real_part
        combined[1::2] = imag_part
        data = combined
    
    data = np.asarray(data, dtype=np.float64)
    mean_val = np.mean(data)
    std_val = np.std(data)
    if std_val > 0:
        normalized = (data - mean_val) / std_val
    else:
        normalized = data - mean_val
    
    clipped = np.clip(normalized, -6.0, 6.0)
    scaled = ((clipped + 6.0) / 12.0) * (2**32 - 1)
    uint32_data = scaled.astype(np.uint32)
    return uint32_data.tobytes()

def squeeze_to_hash_size(data, target_bytes=32):
    """Squeeze byte array to target size using XOR folding"""
    if len(data) <= target_bytes:
        return data + b'\x00' * (target_bytes - len(data))
    
    result = bytearray(target_bytes)
    for i, byte_val in enumerate(data):
        result[i % target_bytes] ^= byte_val
    return bytes(result)

def forward_true_rft(x):
    """Python fallback for RFT"""
    try:
        from core.encryption.resonance_fourier import perform_rft_list
        if len(x) == 0:
            return np.array([], dtype=complex)
        result = perform_rft_list(x.tolist())
        return np.array([complex(freq, amp) for freq, amp in result])
    except:
        n = len(x)
        if n == 0:
            return np.array([], dtype=complex)
        
        phi = (1 + np.sqrt(5)) / 2
        result = []
        for k in range(n):
            amp = sum(x[j] * np.exp(-2j * np.pi * k * j / n) for j in range(n))
            result.append(amp)
        
        return np.array(result, dtype=complex)

def post_rft_mixing(X):
    """Post-RFT mixing"""
    if len(X) == 0:
        return X
    
    mixed = X.copy()
    magnitudes = np.abs(mixed)
    if np.sum(magnitudes) > 0:
        phases = np.angle(mixed)
        new_phases = phases + 0.5 * np.sin(magnitudes * np.pi)
        mixed = magnitudes * np.exp(1j * new_phases)
    
    for i in range(1, len(mixed)):
        mixed[i] += 0.1 * mixed[i-1]
    
    return mixed

def geometric_waveform_hash_direct(msg: bytes, key: bytes, rounds: int = 3) -> bytes:
    """Direct implementation without problematic imports"""
    if len(msg) == 0:
        msg = b'\x00'
    
    # 1) Pre-diffusion  
    m1 = keyed_nonlinear_diffusion(msg, key, rounds=rounds)
    
    # 2) RFT core
    if len(m1) >= 4:
        x = np.frombuffer(m1[:len(m1)//4*4], dtype=np.uint32).astype(np.float64)
    else:
        x = np.array([float(b) for b in m1], dtype=np.float64)
    
    if len(x) == 0:
        x = np.array([1.0], dtype=np.float64)
    
    # Try C++ engine first
    try:
        import quantonium_core
        rft_engine = quantonium_core.ResonanceFourierTransform(x.tolist())
        X = np.array(rft_engine.forward_transform(), dtype=complex)
        print("Using C++ RFT engine")
    except:
        X = forward_true_rft(x)
        print("Using Python RFT fallback")
    
    # 3) Post-mix
    Z = post_rft_mixing(X)
    
    # 4) Pack + post-diffusion
    out = to_packed_bytes(Z)
    out = keyed_nonlinear_diffusion(out, key, rounds=rounds)
    
    # 5) Squeeze
    return squeeze_to_hash_size(out, target_bytes=32)

if __name__ == "__main__":
    from statistics import mean, pstdev
    
    def bit_avalanche_rate(a, b):
        if len(a) != len(b):
            max_len = max(len(a), len(b))
            a = a + b'\x00' * (max_len - len(a))
            b = b + b'\x00' * (max_len - len(b))
        da = int.from_bytes(a, 'little') ^ int.from_bytes(b, 'little')
        bits = len(a) * 8
        return (da.bit_count() / bits) * 100.0

    key = b'test-key-12345678'
    rng = np.random.default_rng(42)
    rates = []

    print('Testing improved hash with rounds=3...')
    for i in range(20):
        if i % 5 == 0: 
            print(f'  {i}/20')
        m = rng.bytes(64)
        h1 = geometric_waveform_hash_direct(m, key, rounds=3)
        b = bytearray(m)
        b[0] ^= 1
        h2 = geometric_waveform_hash_direct(bytes(b), key, rounds=3)
        rates.append(bit_avalanche_rate(h1, h2))
    
    mu = mean(rates)
    sigma = pstdev(rates) if len(rates) > 1 else 0
    print(f'Results: μ={mu:.1f}%, σ={sigma:.1f}%')
    print(f'Hash length: {len(h1)} bytes')
    print(f'Target: μ~50%, σ≤2%')
    print(f'Status: {"✓ PASS" if 48 <= mu <= 52 and sigma <= 2 else "⚠ NEEDS WORK"}')
