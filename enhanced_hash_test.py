"""
Enhanced Geometric Waveform Hash with σ ≤ 2.0 target
Integrates MDS mixing, Weyl keying, and branch-heavy finalizer
"""

import numpy as np
import sys
import os
sys.path.append('.')

# Inline imports to avoid dependency issues
from encryption.diffusion import keyed_nonlinear_diffusion_v2
from encryption.finalizer import finalize_words

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
    """Post-RFT mixing with enhanced nonlinearity"""
    if len(X) == 0:
        return X
    
    mixed = X.copy()
    magnitudes = np.abs(mixed)
    if np.sum(magnitudes) > 0:
        phases = np.angle(mixed)
        # Enhanced nonlinear phase mixing
        new_phases = phases + 0.5 * np.sin(magnitudes * np.pi) + 0.3 * np.cos(magnitudes * 2 * np.pi)
        mixed = magnitudes * np.exp(1j * new_phases)
    
    # Cross-coupling with more diffusion
    for i in range(1, len(mixed)):
        mixed[i] += 0.1 * mixed[i-1]
        if i < len(mixed) - 1:
            mixed[i] += 0.05 * mixed[i+1]  # Forward coupling too
    
    return mixed

def enhanced_geometric_hash(msg: bytes, key: bytes, rounds: int = 3) -> bytes:
    """
    Enhanced hash with MDS mixing, Weyl keying, and finalizer for σ ≤ 2.0
    """
    if len(msg) == 0:
        msg = b'\x00'
    
    # 1) Pre-diffusion with v2 (includes MDS + Weyl)
    m1 = keyed_nonlinear_diffusion_v2(msg, key, rounds=rounds)
    
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
    except:
        X = forward_true_rft(x)
    
    # 3) Enhanced post-mix
    Z = post_rft_mixing(X)
    
    # 4) Pack with finalizer integration
    packed = to_packed_bytes(Z)
    packed_words = np.frombuffer(packed, dtype=np.uint32).copy()
    packed_words = finalize_words(packed_words)
    out = (packed_words.astype(np.uint32)).tobytes()
    
    # 5) Post-diffusion v2 + squeeze
    out = keyed_nonlinear_diffusion_v2(out, key, rounds=rounds)
    return squeeze_to_hash_size(out, target_bytes=32)

def wide_pipe_enhanced_hash(msg: bytes, key: bytes, rounds: int = 3) -> bytes:
    """
    Wide-pipe version for even better σ performance
    """
    if len(msg) == 0:
        msg = b'\x00'
    
    # Generate two states with different counters
    state0_key = key + b'\x00'
    state1_key = key + b'\x01'
    
    # Get two 64-byte states
    state0 = enhanced_geometric_hash(msg, state0_key, rounds)
    state0 = state0 + enhanced_geometric_hash(msg + b'\x00', state0_key, rounds)  # 64 bytes
    
    state1 = enhanced_geometric_hash(msg, state1_key, rounds)  
    state1 = state1 + enhanced_geometric_hash(msg + b'\x01', state1_key, rounds)  # 64 bytes
    
    # Convert to uint32 arrays
    s0_words = np.frombuffer(state0, dtype=np.uint32)
    s1_words = np.frombuffer(state1, dtype=np.uint32)
    
    # Wide-pipe mix
    wide = s0_words ^ np.roll(s1_words, 7)
    
    # Final squeeze to 32 bytes
    return wide.astype(np.uint32).tobytes()[:32]

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
    
    print('Testing enhanced hash with MDS + Weyl + finalizer...')
    
    for hash_fn, name in [(enhanced_geometric_hash, "Enhanced"), (wide_pipe_enhanced_hash, "Wide-pipe")]:
        rates = []
        for i in range(100):
            if i % 25 == 0: 
                print(f'  {name} {i}/100')
            m = rng.bytes(64)
            h1 = hash_fn(m, key, rounds=3)
            
            # Randomize bit position per trial
            b = bytearray(m)
            bit_idx = rng.integers(0, len(b))
            bit_pos = rng.integers(0, 8)
            b[bit_idx] ^= (1 << bit_pos)
            
            h2 = hash_fn(bytes(b), key, rounds=3)
            rates.append(bit_avalanche_rate(h1, h2))
        
        mu = mean(rates)
        sigma = pstdev(rates)
        status = "✓ TARGET ACHIEVED" if 48 <= mu <= 52 and sigma <= 2.0 else "⚠ TUNING"
        print(f'{name}: μ={mu:.1f}%, σ={sigma:.3f}% [{status}]')
    
    print('\nTarget: μ=50±2%, σ≤2.000%')
