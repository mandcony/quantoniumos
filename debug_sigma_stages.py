"""
Micro-ablation tool for debugging σ-tightening
Tracks avalanche at each stage of the hash pipeline
"""

import sys
import os
sys.path.append('.')
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import numpy as np

# Inline bytepacking to avoid import issues
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

from encryption.diffusion import keyed_nonlinear_diffusion

def forward_true_rft(x):
    """Python fallback for RFT when C++ not available"""
    try:
        from core.encryption.resonance_fourier import perform_rft_list
        if len(x) == 0:
            return np.array([], dtype=complex)
        result = perform_rft_list(x.tolist())
        return np.array([complex(freq, amp) for freq, amp in result])
    except ImportError:
        n = len(x)
        if n == 0:
            return np.array([], dtype=complex)
        
        phi = (1 + np.sqrt(5)) / 2
        freqs = np.array([i * phi for i in range(n)])
        
        result = []
        for k, freq in enumerate(freqs):
            amp = sum(x[j] * np.exp(-2j * np.pi * k * j / n) for j in range(n))
            result.append(amp)
        
        return np.array(result, dtype=complex)

def post_rft_mixing(X):
    """Post-RFT mixing for enhanced diffusion"""
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

def bit_avalanche_rate(a: bytes, b: bytes) -> float:
    """Calculate percentage of bits that differ between two byte arrays"""
    if len(a) != len(b):
        max_len = max(len(a), len(b))
        a = a + b'\x00' * (max_len - len(a))
        b = b + b'\x00' * (max_len - len(b))
    
    da = int.from_bytes(a, 'little') ^ int.from_bytes(b, 'little')
    bits = len(a) * 8
    return (da.bit_count() / bits) * 100.0

def debug_hash_stages(msg: bytes, key: bytes, rounds: int = 3):
    """
    Debug version that tracks avalanche at each stage:
    msg -> preDiff -> RFT -> postMix -> pack -> postDiff -> squeeze
    """
    # Create modified message (flip one bit)
    msg_modified = bytearray(msg)
    if len(msg_modified) > 0:
        msg_modified[0] ^= 1
    else:
        msg_modified = bytearray([1])
    msg_modified = bytes(msg_modified)
    
    results = {}
    
    def process_path(input_msg, label):
        stages = {}
        
        # Stage 1: Pre-diffusion
        stages['preDiff'] = keyed_nonlinear_diffusion(input_msg, key, rounds=rounds)
        
        # Stage 2: Convert to numeric and apply RFT
        m1 = stages['preDiff']
        if len(m1) >= 4:
            x = np.frombuffer(m1[:len(m1)//4*4], dtype=np.uint32).astype(np.float64)
        else:
            x = np.array([float(b) for b in m1], dtype=np.float64)
        
        if len(x) == 0:
            x = np.array([1.0], dtype=np.float64)
        
        # Try C++ engine first, fallback to Python
        try:
            import quantonium_core
            rft_engine = quantonium_core.ResonanceFourierTransform(x.tolist())
            X = np.array(rft_engine.forward_transform(), dtype=complex)
        except:
            X = forward_true_rft(x)
        
        stages['RFT'] = X.tobytes()
        
        # Stage 3: Post-mix
        Z = post_rft_mixing(X)
        stages['postMix'] = Z.tobytes()
        
        # Stage 4: Pack
        packed = to_packed_bytes(Z)
        stages['pack'] = packed
        
        # Stage 5: Post-diffusion
        stages['postDiff'] = keyed_nonlinear_diffusion(packed, key, rounds=rounds)
        
        # Stage 6: Squeeze
        stages['squeeze'] = squeeze_to_hash_size(stages['postDiff'], target_bytes=32)
        
        return stages
    
    # Process both original and modified messages
    orig_stages = process_path(msg, 'original')
    mod_stages = process_path(msg_modified, 'modified')
    
    # Calculate avalanche at each stage
    avalanche_results = {}
    stage_names = ['preDiff', 'RFT', 'postMix', 'pack', 'postDiff', 'squeeze']
    
    for stage in stage_names:
        rate = bit_avalanche_rate(orig_stages[stage], mod_stages[stage])
        avalanche_results[stage] = rate
        print(f"{stage:>10}: {rate:6.2f}% bits flipped")
    
    return avalanche_results

def run_micro_ablation_test(num_samples=10):
    """Run micro-ablation test on random samples"""
    rng = np.random.default_rng(42)
    key = b'test-key-1234567'
    
    print("=== MICRO-ABLATION DEBUG ===")
    print("Tracking avalanche at each stage:")
    print()
    
    all_results = []
    
    for i in range(num_samples):
        msg = rng.bytes(64)
        print(f"Sample {i+1}:")
        result = debug_hash_stages(msg, key, rounds=3)
        all_results.append(result)
        print()
    
    # Average results
    print("AVERAGE AVALANCHE BY STAGE:")
    stage_names = ['preDiff', 'RFT', 'postMix', 'pack', 'postDiff', 'squeeze']
    
    for stage in stage_names:
        avg_rate = np.mean([r[stage] for r in all_results])
        print(f"{stage:>10}: {avg_rate:6.2f}% average")
    
    return all_results

if __name__ == "__main__":
    run_micro_ablation_test()
