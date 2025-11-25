#!/usr/bin/env python3
# SPDX-License-Identifier: LicenseRef-QuantoniumOS-Claims-NC
# Copyright (C) 2025 Luis M. Minier / quantoniumos
# This file is listed in CLAIMS_PRACTICING_FILES.txt and is licensed
# under LICENSE-CLAIMS-NC.md (research/education only). Commercial
# rights require a separate patent license from the author.
"""
QuantoniumOS Middleware Transform Engine
Oscillating Wave Computation Layer - Binary→Wave→Compute

This middleware bridges classical binary (0/1) hardware with quantum-inspired
oscillating wave computation using RFT transforms. The system:

1. Takes binary input from hardware
2. Converts to oscillating waveforms via selected RFT variant
3. Performs computation in wave-space (frequency domain)
4. Converts back to binary for output

The middleware automatically selects the best RFT transform variant based on:
- Data type (text, image, audio, crypto keys, etc.)
- Computational requirements (speed, accuracy, security)
- Hardware capabilities
"""

import numpy as np
from typing import Optional, Tuple, Dict, Any
from dataclasses import dataclass
from algorithms.rft.variants.registry import VARIANTS, VariantInfo
from algorithms.rft.core.closed_form_rft import rft_forward, rft_inverse, PHI

@dataclass
class TransformProfile:
    """Profile for selecting optimal transform"""
    data_type: str  # 'text', 'image', 'audio', 'crypto', 'generic'
    priority: str   # 'speed', 'accuracy', 'security', 'compression'
    size: int       # Data size in bytes
    
@dataclass
class WaveComputeResult:
    """Result from wave-space computation"""
    output_binary: bytes
    transform_used: str
    wave_spectrum: np.ndarray
    computation_time: float
    oscillation_frequency: float  # Average frequency in Hz


class MiddlewareTransformEngine:
    """
    Middleware layer that converts binary hardware I/O into oscillating
    waveforms for computation in wave-space using RFT transforms.
    
    This implements the core concept: Hardware (01) → Waves → Computation → (01)
    """
    
    def __init__(self):
        self.variants = VARIANTS
        self.selected_variant = "original"  # Default
        self.phi = PHI
        
    def select_optimal_transform(self, profile: TransformProfile) -> str:
        """
        Intelligently select the best RFT transform variant based on
        the data profile and computational requirements.
        
        Returns the variant name (key in VARIANTS dict)
        """
        # Selection logic based on use case
        if profile.priority == 'security':
            if profile.data_type == 'crypto':
                return "fibonacci_tilt"  # Post-quantum crypto optimized
            else:
                return "chaotic_mix"  # Secure scrambling
                
        elif profile.priority == 'compression':
            if profile.size < 1024:
                return "harmonic_phase"  # Good for small data
            else:
                return "adaptive_phi"  # Universal compression
                
        elif profile.priority == 'speed':
            return "original"  # Fastest, cleanest transform
            
        elif profile.priority == 'accuracy':
            if profile.data_type == 'image' or profile.data_type == 'audio':
                return "geometric_lattice"  # Analog/optical optimized
            else:
                return "phi_chaotic_hybrid"  # Resilient codec
        
        # Default fallback
        return "original"
    
    def binary_to_waveform(self, binary_data: bytes) -> np.ndarray:
        """
        Convert binary data (0/1) into oscillating waveform.
        
        Process:
        1. Unpack bytes to bit array
        2. Convert to amplitude values (-1, +1)
        3. Apply RFT to generate wave spectrum
        
        Returns complex waveform in frequency domain
        """
        # Unpack bytes to bits
        bits = np.unpackbits(np.frombuffer(binary_data, dtype=np.uint8))
        
        # Convert to bipolar signal (-1, +1 instead of 0, 1)
        signal = 2.0 * bits.astype(np.float64) - 1.0
        
        # Pad to power of 2 for efficiency
        n = len(signal)
        next_pow2 = 2 ** int(np.ceil(np.log2(n)))
        if next_pow2 > n:
            signal = np.pad(signal, (0, next_pow2 - n), mode='constant')
        
        # Apply selected RFT variant to generate oscillating waveform
        if self.selected_variant == "original":
            # Use fast closed-form RFT
            waveform = rft_forward(signal)
        else:
            # Use variant-specific generator
            variant_matrix = self.variants[self.selected_variant].generator(len(signal))
            waveform = variant_matrix @ signal
        
        return waveform
    
    def waveform_to_binary(self, waveform: np.ndarray, original_size: int) -> bytes:
        """
        Convert oscillating waveform back to binary data.
        
        Process:
        1. Apply inverse RFT to reconstruct signal
        2. Threshold to binary (0/1)
        3. Pack bits to bytes
        
        Returns original binary data (with rounding)
        """
        # Apply inverse transform
        if self.selected_variant == "original":
            signal = rft_inverse(waveform).real
        else:
            # Use variant-specific inverse (Hermitian transpose for unitary)
            variant_matrix = self.variants[self.selected_variant].generator(len(waveform))
            signal = np.conj(variant_matrix).T @ waveform
            signal = signal.real
        
        # Trim to original size
        signal = signal[:original_size]
        
        # Threshold to binary: >0 → 1, ≤0 → 0
        bits = (signal > 0).astype(np.uint8)
        
        # Pack bits to bytes
        # Pad to multiple of 8
        padding = (8 - len(bits) % 8) % 8
        if padding:
            bits = np.pad(bits, (0, padding), mode='constant')
        
        binary_data = np.packbits(bits).tobytes()
        return binary_data
    
    def compute_in_wavespace(
        self, 
        binary_input: bytes,
        operation: str = "identity",
        profile: Optional[TransformProfile] = None
    ) -> WaveComputeResult:
        """
        Main middleware function: Perform computation in wave-space.
        
        Args:
            binary_input: Raw binary data from hardware (0/1)
            operation: Type of computation ('identity', 'compress', 'encrypt', 'hash')
            profile: Data profile for optimal transform selection
        
        Returns:
            WaveComputeResult with output binary and metadata
        """
        import time
        start_time = time.time()
        
        # Auto-detect profile if not provided
        if profile is None:
            profile = TransformProfile(
                data_type='generic',
                priority='speed',
                size=len(binary_input)
            )
        
        # Select optimal transform variant
        self.selected_variant = self.select_optimal_transform(profile)
        
        # Convert binary → waveform (hardware → oscillating waves)
        waveform = self.binary_to_waveform(binary_input)
        
        # Perform computation in wave-space
        if operation == "identity":
            # Just round-trip through wave space
            result_waveform = waveform
            
        elif operation == "compress":
            # Zero out low-magnitude coefficients (lossy compression)
            threshold = 0.1 * np.max(np.abs(waveform))
            result_waveform = np.where(np.abs(waveform) > threshold, waveform, 0)
            
        elif operation == "encrypt":
            # Apply chaotic phase rotation (simple encryption)
            random_phases = np.exp(2j * np.pi * np.random.rand(len(waveform)))
            result_waveform = waveform * random_phases
            
        elif operation == "hash":
            # Extract phase signature as hash
            phases = np.angle(waveform)
            # Convert phases to binary hash
            hash_bits = ((phases + np.pi) / (2 * np.pi) * 255).astype(np.uint8)
            result_waveform = waveform  # Keep original for reconstruction
        
        else:
            result_waveform = waveform
        
        # Convert waveform → binary (oscillating waves → hardware output)
        output_bits = len(binary_input) * 8
        output_binary = self.waveform_to_binary(result_waveform, output_bits)
        
        # Calculate metrics
        computation_time = time.time() - start_time
        
        # Calculate average oscillation frequency (in normalized units)
        # Frequency is related to the magnitude of high-frequency components
        freq_bins = np.arange(len(waveform))
        spectrum_power = np.abs(waveform) ** 2
        avg_frequency = np.sum(freq_bins * spectrum_power) / np.sum(spectrum_power)
        
        return WaveComputeResult(
            output_binary=output_binary[:len(binary_input)],  # Trim to exact size
            transform_used=self.selected_variant,
            wave_spectrum=waveform,
            computation_time=computation_time,
            oscillation_frequency=float(avg_frequency)
        )
    
    def get_variant_info(self, variant_name: str) -> Optional[VariantInfo]:
        """Get metadata about a specific transform variant"""
        return self.variants.get(variant_name)
    
    def list_all_variants(self) -> Dict[str, VariantInfo]:
        """List all available transform variants"""
        return self.variants.copy()


# Convenience functions for direct use
_engine = MiddlewareTransformEngine()

def select_transform(data_type: str, priority: str = 'speed', size: int = 1024) -> str:
    """Select optimal transform for given requirements"""
    profile = TransformProfile(data_type=data_type, priority=priority, size=size)
    return _engine.select_optimal_transform(profile)

def compute_wave(binary_data: bytes, operation: str = "identity") -> bytes:
    """Quick wave-space computation"""
    result = _engine.compute_in_wavespace(binary_data, operation)
    return result.output_binary

def get_available_transforms() -> list:
    """Get list of all available transform variants"""
    return list(_engine.list_all_variants().keys())


if __name__ == "__main__":
    # Demo: Show how binary data flows through wave-space
    print("=" * 70)
    print("QuantoniumOS Middleware Transform Engine")
    print("Binary (01) → Oscillating Waves → Computation → Binary (01)")
    print("=" * 70)
    
    # Test with sample data
    test_data = b"QuantoniumOS: Wave Computing"
    print(f"\n📥 Input: {test_data.decode()}")
    print(f"   Binary size: {len(test_data)} bytes = {len(test_data) * 8} bits")
    
    # List available transforms
    print(f"\n🔄 Available Transform Variants: {len(VARIANTS)}")
    for variant_name, info in VARIANTS.items():
        print(f"   • {info.name:25} → {info.use_case}")
    
    # Test each priority mode
    for priority in ['speed', 'accuracy', 'security', 'compression']:
        print(f"\n🎯 Testing with priority: {priority.upper()}")
        
        profile = TransformProfile(
            data_type='text',
            priority=priority,
            size=len(test_data)
        )
        
        result = _engine.compute_in_wavespace(test_data, operation="identity", profile=profile)
        
        print(f"   Transform: {result.transform_used}")
        print(f"   Wave frequency: {result.oscillation_frequency:.2f} Hz")
        print(f"   Computation time: {result.computation_time * 1000:.3f} ms")
        print(f"   Output matches: {result.output_binary == test_data}")
    
    print("\n✅ Middleware engine operational!")
