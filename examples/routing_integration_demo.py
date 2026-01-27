#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Luis M. Minier / quantoniumos
"""
RFT Variant Routing Integration Example
========================================

Demonstrates how to use the routing helper with existing codecs.
"""

import numpy as np
import sys
from pathlib import Path

# Add project root
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from algorithms.rft.routing import select_best_variant, detect_signal_type
from algorithms.rft.hybrids.rft_hybrid_codec import RFTHybridCodec


def example_quantum_compression():
    """Example: Quantum state compression with CASCADE variant."""
    print("=" * 70)
    print("EXAMPLE 1: Quantum State Compression")
    print("=" * 70)
    
    # Simulate quantum state (superposition)
    n = 1024
    quantum_state = np.random.randn(n) + 1j * np.random.randn(n)
    quantum_state /= np.linalg.norm(quantum_state)
    
    # Select optimal variant for quantum
    variant = select_best_variant('quantum')
    print(f"Selected variant: {variant} (CASCADE)")
    print(f"Reason: η=0 zero coherence - ideal for quantum superposition")
    print()
    
    # Compress (using real part for demo)
    codec = RFTHybridCodec(mode='h3_cascade', n=n)
    result = codec.encode(quantum_state.real)
    compressed = result['compressed']
    
    print(f"Compression result:")
    print(f"  BPP: {compressed['bpp']:.3f}")
    print(f"  PSNR: {compressed['psnr']:.1f} dB" if compressed['psnr'] else "  PSNR: N/A")
    print(f"  Coherence: {compressed['coherence']:.2e}")
    print()


def example_edge_detection():
    """Example: Step function compression with ENTROPY_GUIDED."""
    print("=" * 70)
    print("EXAMPLE 2: Edge/Step Function Compression")
    print("=" * 70)
    
    # Create step function
    n = 1024
    signal = np.zeros(n)
    signal[n//4:n//2] = 1.0
    signal[3*n//4:] = 0.5
    
    # Detect signal type
    detected_type = detect_signal_type(signal)
    print(f"Detected signal type: {detected_type}")
    
    # Select optimal variant
    variant = select_best_variant('edges')
    print(f"Selected variant: {variant} (ENTROPY_GUIDED)")
    print(f"Reason: 0.406 BPP on steps - 50% improvement over baseline")
    print()
    
    # Compress
    codec = RFTHybridCodec(mode='fh5_entropy', n=n)
    result = codec.encode(signal)
    compressed = result['compressed']
    
    print(f"Compression result:")
    print(f"  BPP: {compressed['bpp']:.3f}")
    print(f"  PSNR: {compressed['psnr']:.1f} dB" if compressed['psnr'] else "  PSNR: N/A")
    print()


def example_audio_mastering():
    """Example: Audio mastering with DICTIONARY variant."""
    print("=" * 70)
    print("EXAMPLE 3: Audio Mastering (High Quality)")
    print("=" * 70)
    
    # Create smooth audio signal
    n = 1024
    t = np.linspace(0, 4*np.pi, n)
    audio = np.sin(t) + 0.3*np.sin(3*t) + 0.1*np.sin(5*t)
    
    # Detect signal type
    detected_type = detect_signal_type(audio)
    print(f"Detected signal type: {detected_type}")
    
    # Select for quality
    variant = select_best_variant('audio', quality_target='quality')
    print(f"Selected variant: {variant} (DICTIONARY)")
    print(f"Reason: 49.9 dB PSNR - maximum quality for smooth signals")
    print()
    
    # Compress
    codec = RFTHybridCodec(mode='h6_dictionary', n=n)
    result = codec.encode(audio)
    compressed = result['compressed']
    
    print(f"Compression result:")
    print(f"  BPP: {compressed['bpp']:.3f}")
    print(f"  PSNR: {compressed['psnr']:.1f} dB" if compressed['psnr'] else "  PSNR: N/A")
    print()


def example_auto_detection():
    """Example: Automatic signal type detection."""
    print("=" * 70)
    print("EXAMPLE 4: Automatic Signal Type Detection")
    print("=" * 70)
    
    test_signals = {
        'Smooth sine': np.sin(np.linspace(0, 4*np.pi, 1024)),
        'Step function': np.concatenate([np.zeros(512), np.ones(512)]),
        'Random noise': np.random.randn(1024),
        'Harmonic': np.sin(np.linspace(0, 10*np.pi, 1024)) + \
                    0.5*np.sin(2*np.linspace(0, 10*np.pi, 1024)),
    }
    
    for name, signal in test_signals.items():
        detected = detect_signal_type(signal)
        variant = select_best_variant('auto', signal=signal)
        
        from algorithms.rft.routing import VARIANT_INFO
        variant_name = VARIANT_INFO[variant]['name']
        
        print(f"{name:>20}: {detected:>10} → {variant_name:>18} (ID {variant})")
    
    print()


if __name__ == '__main__':
    example_quantum_compression()
    example_edge_detection()
    example_audio_mastering()
    example_auto_detection()
    
    print("=" * 70)
    print("Integration complete! Use routing.py for optimal variant selection.")
    print("=" * 70)
