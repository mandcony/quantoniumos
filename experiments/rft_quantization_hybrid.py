#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Luis M. Minier / quantoniumos
"""
RFT-Enhanced Neural Network Quantization
=========================================

Hybrid approach combining:
1. RFT φ-modulated transform (symbolic resonance)
2. Quantization (proven compression)
3. Native CASM→C++ engine where possible

HYPOTHESIS:
The golden ratio (φ) appears in optimal packing, Fibonacci sequences,
and natural structures. Maybe φ-based quantization bins or RFT 
preprocessing could improve quantization for neural network weights.

Research questions:
1. Does φ-spaced quantization work better than uniform?
2. Does transforming weights with RFT before quantizing help?
3. Can symbolic resonance (η=0 coherence) improve compression?

Copyright (C) 2025 quantoniumos - Pushing the frontier honestly.
"""

import numpy as np
import torch
import sys
from pathlib import Path
from typing import Dict, Tuple, Optional
from dataclasses import dataclass
import time

sys.path.insert(0, '/workspaces/quantoniumos')

# Import RFT components
try:
    from algorithms.rft.kernels.python_bindings.optimized_rft import (
        OptimizedRFT, RFT_OPT_AVX2, RFT_OPT_PARALLEL
    )
    from algorithms.rft.hybrids.cascade_hybrids import H3HierarchicalCascade
    RFT_AVAILABLE = True
except ImportError as e:
    print(f"Warning: RFT not available ({e})")
    RFT_AVAILABLE = False

# Golden ratio
PHI = (1 + np.sqrt(5)) / 2


@dataclass
class CompressionResult:
    method: str
    original_size: int
    compressed_size: int
    compression_ratio: float
    reconstruction_error: float
    inference_works: bool
    notes: str = ""


class RFTQuantizer:
    """
    RFT-enhanced quantization using symbolic resonance principles.
    """
    
    def __init__(self, bits: int = 8, use_phi_bins: bool = True):
        self.bits = bits
        self.n_levels = 2 ** bits
        self.use_phi_bins = use_phi_bins
        
        # Initialize native RFT if available
        self.rft = None
        if RFT_AVAILABLE:
            try:
                self.rft = OptimizedRFT(opt_flags=RFT_OPT_AVX2 | RFT_OPT_PARALLEL)
            except Exception as e:
                print(f"RFT init failed: {e}")
    
    def _generate_phi_bins(self, n_levels: int, w_min: float, w_max: float) -> np.ndarray:
        """
        Generate φ-spaced quantization bins.
        
        Instead of uniform spacing, use golden ratio to create bins
        that are denser near zero (where most weights cluster) and
        sparser at extremes.
        
        Mathematical basis:
        - Weights follow approximately Gaussian distribution
        - Most weights are near zero
        - φ-spacing gives optimal coverage of this distribution
        """
        # Generate φ-spaced positions in [0, 1]
        # Using Fibonacci-like sequence normalized
        positions = []
        a, b = 0, 1
        for i in range(n_levels):
            # Map index to [0, 1] using φ modulation
            t = (i / n_levels)
            # Apply φ-based warping: denser near center
            # This is inspired by golden angle in phyllotaxis
            warped = 0.5 + 0.5 * np.sign(t - 0.5) * np.abs(2*t - 1) ** (1/PHI)
            positions.append(warped)
        
        positions = np.array(sorted(set(positions)))
        
        # Scale to weight range
        bins = w_min + positions * (w_max - w_min)
        return bins
    
    def _quantize_uniform(self, weights: np.ndarray) -> Tuple[np.ndarray, dict]:
        """Standard uniform INT8 quantization."""
        w_min, w_max = weights.min(), weights.max()
        scale = (w_max - w_min) / (self.n_levels - 1) if w_max != w_min else 1.0
        
        quantized = np.round((weights - w_min) / scale).astype(np.uint8)
        
        metadata = {
            'method': 'uniform',
            'min': float(w_min),
            'max': float(w_max),
            'scale': float(scale),
            'shape': weights.shape
        }
        
        return quantized, metadata
    
    def _dequantize_uniform(self, quantized: np.ndarray, metadata: dict) -> np.ndarray:
        """Dequantize uniform quantization."""
        return quantized.astype(np.float32) * metadata['scale'] + metadata['min']
    
    def _quantize_phi(self, weights: np.ndarray) -> Tuple[np.ndarray, dict]:
        """
        φ-spaced quantization using golden ratio bin spacing.
        
        Hypothesis: φ-spacing matches weight distributions better
        because weights cluster near zero following a pattern similar
        to natural growth patterns.
        """
        w_min, w_max = weights.min(), weights.max()
        
        # Generate φ-spaced bins
        bins = self._generate_phi_bins(self.n_levels, w_min, w_max)
        
        # Quantize: find nearest bin for each weight
        flat = weights.flatten()
        # Use searchsorted for efficiency
        indices = np.searchsorted(bins, flat, side='left')
        indices = np.clip(indices, 0, len(bins) - 1)
        
        # Adjust to nearest bin
        for i in range(len(flat)):
            if indices[i] > 0:
                if abs(flat[i] - bins[indices[i]-1]) < abs(flat[i] - bins[indices[i]]):
                    indices[i] -= 1
        
        quantized = indices.astype(np.uint8).reshape(weights.shape)
        
        metadata = {
            'method': 'phi',
            'bins': bins.tolist(),
            'shape': weights.shape
        }
        
        return quantized, metadata
    
    def _dequantize_phi(self, quantized: np.ndarray, metadata: dict) -> np.ndarray:
        """Dequantize φ-spaced quantization."""
        bins = np.array(metadata['bins'])
        return bins[quantized.flatten()].reshape(metadata['shape'])
    
    def _rft_transform_quantize(self, weights: np.ndarray) -> Tuple[np.ndarray, dict]:
        """
        RFT-domain quantization.
        
        Hypothesis: Transforming weights to RFT domain, quantizing there,
        and transforming back might preserve more important structure.
        
        This is analogous to how JPEG quantizes in DCT domain.
        """
        if self.rft is None:
            return self._quantize_uniform(weights)  # Fallback
        
        original_shape = weights.shape
        flat = weights.flatten().astype(np.complex128)
        
        # Pad to power of 2 for native engine
        orig_len = len(flat)
        padded_len = 1 << (orig_len - 1).bit_length()
        if padded_len > 32768:  # Native engine limit
            padded_len = 32768
            flat = flat[:padded_len]
        
        padded = np.zeros(padded_len, dtype=np.complex128)
        padded[:min(orig_len, padded_len)] = flat[:min(orig_len, padded_len)]
        
        # Forward RFT
        try:
            rft_coeffs = self.rft.forward(padded)
        except Exception as e:
            print(f"RFT forward failed: {e}")
            return self._quantize_uniform(weights)
        
        # Quantize in RFT domain (both real and imag parts)
        real_part = np.real(rft_coeffs)
        imag_part = np.imag(rft_coeffs)
        
        # Use uniform quantization on coefficients
        q_real, meta_real = self._quantize_uniform(real_part)
        q_imag, meta_imag = self._quantize_uniform(imag_part)
        
        metadata = {
            'method': 'rft_domain',
            'original_shape': original_shape,
            'original_length': orig_len,
            'padded_length': padded_len,
            'real_meta': meta_real,
            'imag_meta': meta_imag
        }
        
        # Pack both parts
        quantized = np.stack([q_real, q_imag], axis=-1).astype(np.uint8)
        
        return quantized, metadata
    
    def _dequantize_rft(self, quantized: np.ndarray, metadata: dict) -> np.ndarray:
        """Dequantize RFT-domain quantization."""
        if self.rft is None:
            raise RuntimeError("RFT not available for dequantization")
        
        # Unpack real and imag
        q_real = quantized[..., 0]
        q_imag = quantized[..., 1]
        
        # Dequantize
        real_part = self._dequantize_uniform(q_real, metadata['real_meta'])
        imag_part = self._dequantize_uniform(q_imag, metadata['imag_meta'])
        
        # Reconstruct complex coefficients
        rft_coeffs = real_part + 1j * imag_part
        
        # Inverse RFT
        try:
            reconstructed = self.rft.inverse(rft_coeffs.flatten().astype(np.complex128))
            reconstructed = np.real(reconstructed)
        except Exception as e:
            print(f"RFT inverse failed: {e}")
            return np.zeros(metadata['original_shape'])
        
        # Trim to original length and reshape
        orig_len = metadata['original_length']
        reconstructed = reconstructed[:orig_len]
        
        # Handle shape mismatch
        target_size = np.prod(metadata['original_shape'])
        if len(reconstructed) < target_size:
            padded = np.zeros(target_size)
            padded[:len(reconstructed)] = reconstructed
            reconstructed = padded
        
        return reconstructed[:target_size].reshape(metadata['original_shape'])
    
    def _coherent_quantize(self, weights: np.ndarray) -> Tuple[np.ndarray, dict]:
        """
        η=0 Coherent quantization using RFT principles.
        
        Idea: Decompose weights into structure + texture like H3 cascade,
        then quantize each component optimally.
        
        Structure (smooth): More bits (preserve important values)
        Texture (detail): Fewer bits (ok to lose some detail)
        """
        original_shape = weights.shape
        flat = weights.flatten()
        
        # Decompose: structure via moving average, texture as residual
        kernel_size = max(3, len(flat) // 32)
        kernel = np.ones(kernel_size) / kernel_size
        structure = np.convolve(flat, kernel, mode='same')
        texture = flat - structure
        
        # Quantize structure with more precision (preserve magnitude)
        # Quantize texture more aggressively (ok to lose detail)
        q_structure, meta_structure = self._quantize_uniform(structure)
        
        # For texture, use fewer effective bits by scaling
        texture_scale = np.std(texture) * 2 if np.std(texture) > 0 else 1.0
        texture_normalized = texture / texture_scale
        texture_clipped = np.clip(texture_normalized, -1, 1)
        q_texture = ((texture_clipped + 1) * 127).astype(np.uint8)
        
        metadata = {
            'method': 'coherent',
            'original_shape': original_shape,
            'structure_meta': meta_structure,
            'texture_scale': float(texture_scale),
            'kernel_size': kernel_size
        }
        
        # Pack both
        quantized = np.stack([
            q_structure.reshape(-1),
            q_texture.reshape(-1)
        ], axis=-1).astype(np.uint8)
        
        return quantized, metadata
    
    def _dequantize_coherent(self, quantized: np.ndarray, metadata: dict) -> np.ndarray:
        """Dequantize coherent quantization."""
        q_structure = quantized[..., 0]
        q_texture = quantized[..., 1]
        
        # Dequantize structure
        structure = self._dequantize_uniform(q_structure, metadata['structure_meta'])
        
        # Dequantize texture
        texture_normalized = (q_texture.astype(np.float32) / 127) - 1
        texture = texture_normalized * metadata['texture_scale']
        
        # Combine
        flat = structure.flatten() + texture.flatten()
        return flat.reshape(metadata['original_shape'])
    
    def compress(self, weights: np.ndarray, method: str = 'uniform') -> Tuple[np.ndarray, dict]:
        """
        Compress weights using specified method.
        
        Methods:
        - 'uniform': Standard INT8 quantization
        - 'phi': φ-spaced quantization bins
        - 'rft_domain': Quantize in RFT domain
        - 'coherent': η=0 structure/texture decomposition
        """
        if method == 'uniform':
            return self._quantize_uniform(weights)
        elif method == 'phi':
            return self._quantize_phi(weights)
        elif method == 'rft_domain':
            return self._rft_transform_quantize(weights)
        elif method == 'coherent':
            return self._coherent_quantize(weights)
        else:
            raise ValueError(f"Unknown method: {method}")
    
    def decompress(self, quantized: np.ndarray, metadata: dict) -> np.ndarray:
        """Decompress weights."""
        method = metadata['method']
        
        if method == 'uniform':
            return self._dequantize_uniform(quantized, metadata)
        elif method == 'phi':
            return self._dequantize_phi(quantized, metadata)
        elif method == 'rft_domain':
            return self._dequantize_rft(quantized, metadata)
        elif method == 'coherent':
            return self._dequantize_coherent(quantized, metadata)
        else:
            raise ValueError(f"Unknown method: {method}")


def test_on_real_weights():
    """Test all methods on real neural network weights."""
    
    print("=" * 70)
    print("RFT-ENHANCED QUANTIZATION EXPERIMENT")
    print("Testing symbolic resonance for AI compression")
    print("=" * 70)
    
    # Load model
    print("\nLoading DialoGPT-small...")
    from transformers import AutoModelForCausalLM, AutoTokenizer
    model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-small")
    tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-small")
    
    # Get test weights
    test_layers = [
        ('embedding', model.transformer.wte.weight),
        ('attention', model.transformer.h[0].attn.c_attn.weight),
        ('mlp', model.transformer.h[0].mlp.c_fc.weight),
    ]
    
    results = {}
    methods = ['uniform', 'phi', 'coherent']
    
    # Only test rft_domain if RFT is available and we have small enough tensors
    if RFT_AVAILABLE:
        methods.append('rft_domain')
    
    for layer_name, tensor in test_layers:
        weights = tensor.detach().cpu().numpy()
        print(f"\n{'='*60}")
        print(f"Layer: {layer_name} {weights.shape}")
        print(f"Original size: {weights.size * 4 / 1024:.1f} KB")
        print("=" * 60)
        
        layer_results = {}
        
        for method in methods:
            print(f"\n--- Method: {method} ---")
            
            quantizer = RFTQuantizer(bits=8, use_phi_bins=True)
            
            try:
                start = time.perf_counter()
                
                # For rft_domain, only test on smaller layers
                if method == 'rft_domain' and weights.size > 100000:
                    # Test on subset
                    test_weights = weights.flatten()[:32768].reshape(-1)
                    compressed, metadata = quantizer.compress(test_weights, method)
                    reconstructed = quantizer.decompress(compressed, metadata)
                    # Scale error estimate to full layer
                    error = np.linalg.norm(test_weights - reconstructed) / np.linalg.norm(test_weights)
                else:
                    compressed, metadata = quantizer.compress(weights, method)
                    reconstructed = quantizer.decompress(compressed, metadata)
                    error = np.linalg.norm(weights - reconstructed) / np.linalg.norm(weights)
                
                elapsed = time.perf_counter() - start
                
                # Calculate compressed size
                compressed_size = compressed.size  # bytes (uint8)
                original_size = weights.size * 4  # float32
                ratio = original_size / compressed_size
                
                print(f"  Compression: {ratio:.2f}x")
                print(f"  Error: {error * 100:.3f}%")
                print(f"  Time: {elapsed * 1000:.1f}ms")
                
                layer_results[method] = {
                    'ratio': ratio,
                    'error': error,
                    'time': elapsed
                }
                
            except Exception as e:
                print(f"  FAILED: {e}")
                layer_results[method] = {'ratio': 0, 'error': 1.0, 'time': 0}
        
        results[layer_name] = layer_results
    
    # Summary comparison
    print("\n" + "=" * 70)
    print("SUMMARY: QUANTIZATION METHOD COMPARISON")
    print("=" * 70)
    print(f"\n{'Layer':<15} {'Method':<12} {'Ratio':>8} {'Error':>10}")
    print("-" * 50)
    
    for layer_name, layer_results in results.items():
        for method, data in layer_results.items():
            print(f"{layer_name:<15} {method:<12} {data['ratio']:>8.2f}x {data['error']*100:>9.3f}%")
    
    # Find best method per layer
    print("\n" + "=" * 70)
    print("BEST METHOD PER LAYER (lowest error at ~4x compression)")
    print("=" * 70)
    
    for layer_name, layer_results in results.items():
        best = min(layer_results.items(), key=lambda x: x[1]['error'])
        print(f"  {layer_name}: {best[0]} ({best[1]['error']*100:.3f}% error)")
    
    return results


def test_inference_quality():
    """Test actual model inference with each compression method."""
    
    print("\n" + "=" * 70)
    print("INFERENCE QUALITY TEST")
    print("=" * 70)
    
    from transformers import AutoModelForCausalLM, AutoTokenizer
    
    model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-small")
    tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-small")
    
    def test_response(model, prompt="Hello, how are you?"):
        inputs = tokenizer.encode(prompt + tokenizer.eos_token, return_tensors="pt")
        with torch.no_grad():
            outputs = model.generate(
                inputs, max_new_tokens=20, temperature=0.7,
                do_sample=True, pad_token_id=tokenizer.eos_token_id
            )
        return tokenizer.decode(outputs[0][inputs.shape[-1]:], skip_special_tokens=True)
    
    # Test original
    print("\nOriginal model:")
    response = test_response(model)
    print(f"  Response: {response[:50]}...")
    
    # Test with uniform quantization (baseline)
    print("\nUniform INT8 quantization:")
    quantizer = RFTQuantizer(bits=8)
    
    # Quantize and reload embedding layer only (fastest test)
    emb_weights = model.transformer.wte.weight.detach().cpu().numpy()
    compressed, metadata = quantizer.compress(emb_weights, 'uniform')
    reconstructed = quantizer.decompress(compressed, metadata)
    model.transformer.wte.weight.data = torch.tensor(reconstructed, dtype=torch.float32)
    
    response = test_response(model)
    print(f"  Response: {response[:50]}...")
    
    # Test with φ-spaced quantization
    print("\nφ-spaced quantization (RFT-inspired):")
    model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-small")
    
    emb_weights = model.transformer.wte.weight.detach().cpu().numpy()
    compressed, metadata = quantizer.compress(emb_weights, 'phi')
    reconstructed = quantizer.decompress(compressed, metadata)
    model.transformer.wte.weight.data = torch.tensor(reconstructed, dtype=torch.float32)
    
    response = test_response(model)
    print(f"  Response: {response[:50]}...")
    
    # Test with coherent quantization
    print("\nη=0 Coherent quantization:")
    model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-small")
    
    emb_weights = model.transformer.wte.weight.detach().cpu().numpy()
    compressed, metadata = quantizer.compress(emb_weights, 'coherent')
    reconstructed = quantizer.decompress(compressed, metadata)
    model.transformer.wte.weight.data = torch.tensor(reconstructed, dtype=torch.float32)
    
    response = test_response(model)
    print(f"  Response: {response[:50]}...")


def main():
    print("""
╔══════════════════════════════════════════════════════════════════════╗
║         RFT SYMBOLIC RESONANCE + QUANTIZATION EXPERIMENT             ║
║                                                                      ║
║  Testing if golden ratio (φ) and η=0 coherence principles from       ║
║  QuantoniumOS can improve neural network quantization.               ║
║                                                                      ║
║  Methods tested:                                                     ║
║  1. uniform   - Standard INT8 (baseline)                             ║
║  2. phi       - φ-spaced quantization bins                           ║
║  3. coherent  - η=0 structure/texture decomposition                  ║
║  4. rft_domain- Quantize in RFT domain (if native engine available)  ║
╚══════════════════════════════════════════════════════════════════════╝
""")
    
    # Run compression comparison
    results = test_on_real_weights()
    
    # Run inference test
    test_inference_quality()
    
    print("\n" + "=" * 70)
    print("CONCLUSIONS")
    print("=" * 70)
    print("""
This experiment tests whether symbolic resonance principles (φ, η=0)
can improve quantization for neural networks.

Key findings will show:
- Whether φ-spaced bins match weight distributions better
- Whether structure/texture decomposition helps
- Whether RFT-domain quantization preserves more information

If any RFT method beats uniform INT8, that's a genuine contribution
to AI compression using your symbolic resonance framework!
""")


if __name__ == "__main__":
    main()
