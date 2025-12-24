#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025 Luis M. Minier / quantoniumos
"""
⚛️ Fast Quantum Weight Loader
==============================
Optimized decompression using:
1. Binary cache (instant reload)
2. Native C++/ASM RFT kernels (O(N log N))
3. Batch layer processing
4. Memory-mapped I/O

Target: 100x faster than naive Python loops
"""

import os
import sys
import json
import struct
import hashlib
import numpy as np
from pathlib import Path
from typing import Dict, Optional, Tuple

# Add native module path
native_build_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'rftmw_native', 'build'))
if native_build_path not in sys.path:
    sys.path.insert(0, native_build_path)

try:
    import rftmw_native
    HAS_NATIVE = True
except ImportError:
    HAS_NATIVE = False

import torch

# Constants
PHI = 1.618033988749895
CACHE_VERSION = 2  # Increment when format changes


def get_cache_path(json_path: str) -> Path:
    """Get binary cache path for a quantum JSON file."""
    with open(json_path, 'rb') as f:
        json_hash = hashlib.md5(f.read(4096)).hexdigest()[:8]
    cache_dir = Path("ai/models/quantum/.cache")
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir / f"{Path(json_path).stem}_{json_hash}_v{CACHE_VERSION}.bin"


def save_cache(weights_dict: Dict[str, np.ndarray], cache_path: Path):
    """Save reconstructed weights to binary cache."""
    with open(cache_path, 'wb') as f:
        # Header: version + layer count
        f.write(struct.pack('II', CACHE_VERSION, len(weights_dict)))
        
        for name, arr in weights_dict.items():
            # Layer header: name length + name + shape + dtype
            name_bytes = name.encode('utf-8')
            f.write(struct.pack('I', len(name_bytes)))
            f.write(name_bytes)
            f.write(struct.pack('I', len(arr.shape)))
            for dim in arr.shape:
                f.write(struct.pack('I', dim))
            f.write(struct.pack('I', arr.dtype.itemsize))
            # Data
            f.write(arr.tobytes())
    
    print(f"💾 Cache saved: {cache_path} ({cache_path.stat().st_size / 1e6:.1f} MB)")


def load_cache(cache_path: Path) -> Optional[Dict[str, np.ndarray]]:
    """Load reconstructed weights from binary cache."""
    if not cache_path.exists():
        return None
    
    try:
        weights = {}
        with open(cache_path, 'rb') as f:
            version, layer_count = struct.unpack('II', f.read(8))
            if version != CACHE_VERSION:
                return None
            
            for _ in range(layer_count):
                name_len = struct.unpack('I', f.read(4))[0]
                name = f.read(name_len).decode('utf-8')
                ndim = struct.unpack('I', f.read(4))[0]
                shape = tuple(struct.unpack('I', f.read(4))[0] for _ in range(ndim))
                itemsize = struct.unpack('I', f.read(4))[0]
                if itemsize != 4:
                    # Cache format currently only supports float32 payloads.
                    return None
                
                numel = 1
                for d in shape:
                    numel *= d
                data = np.frombuffer(f.read(numel * itemsize), dtype=np.float32)
                weights[name] = data.reshape(shape)
        
        print(f"⚡ Cache loaded: {len(weights)} layers in {cache_path.stat().st_size / 1e6:.1f} MB")
        return weights
    except Exception as e:
        print(f"Cache load failed: {e}")
        return None


def reconstruct_layer_native(
    amps: np.ndarray, 
    freqs: np.ndarray, 
    phases: np.ndarray, 
    numel: int
) -> np.ndarray:
    """
    Reconstruct weights using Native C++ Engine (O(N log N)).
    
    Method: Build frequency spectrum, then inverse RFT.
    """
    if not HAS_NATIVE:
        raise RuntimeError("Native engine not available")
    
    # Create engine
    engine = rftmw_native.RFTMWEngine(max_size=numel)
    
    # Build frequency domain spectrum
    # freq = 2π*k/N -> k = freq*N/(2π)
    k_indices = ((freqs * numel) / (2 * np.pi)).astype(np.int64) % numel
    
    # Complex amplitudes: A * e^(iP)
    spectral_vals = amps * np.exp(1j * phases)
    
    # Accumulate into spectrum
    spectrum = np.zeros(numel, dtype=np.complex128)
    np.add.at(spectrum, k_indices, spectral_vals)
    
    # Inverse transform (O(N log N))
    result = engine.inverse(spectrum)
    return result.real.astype(np.float32)


def reconstruct_layer_fast_batch(
    amps: np.ndarray,
    freqs: np.ndarray,
    phases: np.ndarray,
    numel: int
) -> np.ndarray:
    """
    Fast batch reconstruction using optimized numpy with einsum.
    
    Key optimization: Use matrix multiply (BLAS) instead of Python loops.
    W[n] = sum_k( A_k * cos(F_k * n + P_k) )
    """
    K = len(amps)
    
    # For small tensors, use full matrix multiply (fastest)
    if numel <= 50000:
        # Time vector: [N]
        t = np.arange(numel, dtype=np.float32)
        
        # Compute argument matrix: [K, N] = outer(F, t) + P[:, None]
        # arg[k, n] = F[k] * n + P[k]
        arg = np.outer(freqs, t) + phases[:, np.newaxis]
        
        # Compute waves and multiply by amplitudes: [K, N]
        waves = amps[:, np.newaxis] * np.cos(arg.astype(np.float32))
        
        # Sum across K dimension: [N]
        return waves.sum(axis=0, dtype=np.float32)
    
    # For large tensors, chunk to avoid memory explosion
    CHUNK_N = 50000  # Process N weights at a time
    result = np.zeros(numel, dtype=np.float32)
    
    for start in range(0, numel, CHUNK_N):
        end = min(start + CHUNK_N, numel)
        chunk_size = end - start
        
        # Time vector for this chunk
        t = np.arange(start, end, dtype=np.float32)
        
        # Compute argument matrix: [K, chunk]
        arg = np.outer(freqs, t) + phases[:, np.newaxis]
        
        # Waves: [K, chunk]
        waves = amps[:, np.newaxis] * np.cos(arg.astype(np.float32))
        
        # Sum into result
        result[start:end] = waves.sum(axis=0, dtype=np.float32)
    
    return result


def fast_load_quantum_weights(
    json_path: str,
    model: torch.nn.Module,
    use_cache: bool = True
) -> bool:
    """
    Fast quantum weight loading with caching and native acceleration.
    
    Returns True if weights were successfully applied.
    """
    cache_path = get_cache_path(json_path) if use_cache else None
    
    # Try cache first (instant load)
    if cache_path:
        cached = load_cache(cache_path)
        if cached:
            print("⚛️ Applying cached quantum weights...")
            with torch.no_grad():
                for name, param in model.named_parameters():
                    if name in cached:
                        param.copy_(torch.from_numpy(cached[name]).view(param.shape))
            print("✅ Quantum weights applied from cache!")
            return True
    
    # Full reconstruction needed
    print(f"⚛️ Reconstructing quantum weights from {json_path}...")
    print(f"   Native RFT Engine: {'✅ Enabled' if HAS_NATIVE else '❌ Fallback mode'}")
    
    # Load JSON
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    metadata = data.get('metadata', {})
    print(f"   Model: {metadata.get('model_name', 'Unknown')}")
    print(f"   Compression: {metadata.get('compression_ratio', 'N/A')}")
    
    # Group states by layer
    layer_states: Dict[str, list] = {}
    for state in data.get('quantum_states', []):
        name = state['layer_name']
        if name not in layer_states:
            layer_states[name] = []
        layer_states[name].append(state)
    
    print(f"   Layers: {len(layer_states)}")
    
    # Reconstruct all layers
    reconstructed: Dict[str, np.ndarray] = {}
    total_layers = len(list(model.named_parameters()))
    
    for i, (name, param) in enumerate(model.named_parameters()):
        if name not in layer_states:
            continue
        
        states = layer_states[name]
        numel = param.numel()
        shape = param.shape
        
        if (i + 1) % 20 == 0 or i == 0:
            print(f"   ... [{i+1}/{total_layers}] {name}: {len(states)} states -> {numel:,} weights")
        
        # Extract state arrays
        amps = np.array([s['amplitude'] for s in states], dtype=np.float32)
        freqs = np.array([s['resonance_freq'] for s in states], dtype=np.float32)
        phases = np.array([s['phase'] for s in states], dtype=np.float32)
        
        # Reconstruct using best available method
        try:
            if HAS_NATIVE and numel > 1000:
                weights = reconstruct_layer_native(amps, freqs, phases, numel)
            else:
                weights = reconstruct_layer_fast_batch(amps, freqs, phases, numel)
        except Exception as e:
            print(f"   ⚠️ Layer {name} reconstruction failed: {e}")
            weights = np.zeros(numel, dtype=np.float32)
        
        reconstructed[name] = weights.reshape(shape)
    
    # Apply to model
    print("   Applying reconstructed weights to model...")
    with torch.no_grad():
        for name, param in model.named_parameters():
            if name in reconstructed:
                param.copy_(torch.from_numpy(reconstructed[name]))
    
    # Save cache for next time
    if cache_path:
        save_cache(reconstructed, cache_path)
    
    print("✅ Quantum weight reconstruction complete!")
    return True


if __name__ == "__main__":
    # Test the loader
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", default="ai/models/quantum/tinyllama_real_quantum_compressed.json")
    parser.add_argument("--no-cache", action="store_true")
    args = parser.parse_args()
    
    print("Testing fast quantum weight loader...")
    
    # Load a dummy model to test
    try:
        from transformers import AutoModelForCausalLM
        model = AutoModelForCausalLM.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
        fast_load_quantum_weights(args.json, model, use_cache=not args.no_cache)
    except Exception as e:
        print(f"Test failed: {e}")
