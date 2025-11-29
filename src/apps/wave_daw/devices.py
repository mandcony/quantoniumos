#!/usr/bin/env python3
# SPDX-License-Identifier: LicenseRef-QuantoniumOS-Claims-NC
# Copyright (C) 2025 Luis M. Minier / quantoniumos
"""Wave DAW - Built-in Devices

RFT-native audio devices: EQ, filters, dynamics, utilities.
These are the "special sauce" - DSP that uses Φ-RFT transforms.
"""

from typing import Dict, Any, Callable
import numpy as np
from .engine import (
    DeviceNode, DeviceKind, DevicePort, 
    WaveField, rft_forward, rft_inverse, rft_overlap_add,
    PHI, PHI_INV
)


# ═══════════════════════════════════════════════════════════════════════════════
# DEVICE FACTORY
# ═══════════════════════════════════════════════════════════════════════════════

def create_device(subtype: str, name: str = None) -> DeviceNode:
    """Factory function to create devices by type"""
    
    factories = {
        'utility': create_utility,
        'rft_eq': create_rft_eq,
        'rft_filter': create_rft_filter,
        'rft_morph': create_rft_morph,
        'compressor': create_compressor,
        'rft_reverb': create_rft_reverb,
        'meter': create_meter,
    }
    
    if subtype not in factories:
        raise ValueError(f"Unknown device type: {subtype}")
    
    device = factories[subtype]()
    if name:
        device.name = name
    
    return device


# ═══════════════════════════════════════════════════════════════════════════════
# UTILITY DEVICE
# ═══════════════════════════════════════════════════════════════════════════════

def create_utility() -> DeviceNode:
    """Simple gain/pan utility"""
    
    def process(buffer: np.ndarray, params: Dict, state: Dict,
                sample_rate: float, block_size: int) -> np.ndarray:
        gain_db = params.get('gain_db', 0.0)
        pan = params.get('pan', 0.0)
        phase_invert = params.get('phase_invert', False)
        
        # Apply gain
        gain = 10 ** (gain_db / 20)
        output = buffer * gain
        
        # Apply pan
        if buffer.shape[0] >= 2:
            angle = (pan + 1) * np.pi / 4
            output[0] *= np.cos(angle)
            output[1] *= np.sin(angle)
        
        # Phase invert
        if phase_invert:
            output = -output
        
        return output
    
    device = DeviceNode(
        kind=DeviceKind.UTILITY,
        subtype='utility',
        name='Utility',
        params={'gain_db': 0.0, 'pan': 0.0, 'phase_invert': False},
        param_specs={
            'gain_db': {'min': -48, 'max': 12, 'default': 0, 'unit': 'dB'},
            'pan': {'min': -1, 'max': 1, 'default': 0, 'unit': ''},
            'phase_invert': {'min': 0, 'max': 1, 'default': 0, 'unit': 'bool'},
        },
        inputs=[DevicePort('audio_in', 'audio', 2)],
        outputs=[DevicePort('audio_out', 'audio', 2)]
    )
    device._processor = process
    return device


# ═══════════════════════════════════════════════════════════════════════════════
# RFT RESONANT EQ - Signature Φ-spaced bands
# ═══════════════════════════════════════════════════════════════════════════════

def create_rft_eq() -> DeviceNode:
    """
    Φ-RFT Resonant EQ.
    Uses golden-ratio spaced frequency bands for more natural resonance.
    
    Bands are centered at: f0 * PHI^n for n = -2, -1, 0, 1, 2, 3
    """
    
    def process(buffer: np.ndarray, params: Dict, state: Dict,
                sample_rate: float, block_size: int) -> np.ndarray:
        
        # RFT processing function
        def rft_eq_process(coeffs: np.ndarray) -> np.ndarray:
            channels, n = coeffs.shape
            
            # Base frequency and band gains
            base_freq = params.get('base_freq', 440.0)
            band_gains = [
                params.get(f'band_{i}_db', 0.0) for i in range(6)
            ]
            q_factor = params.get('q', 1.0)
            
            freqs = np.fft.fftfreq(n, 1/sample_rate)
            
            for ch in range(channels):
                for i, gain_db in enumerate(band_gains):
                    if abs(gain_db) < 0.1:
                        continue
                    
                    # Φ-spaced center frequency
                    center = base_freq * (PHI ** (i - 2))
                    
                    # Bell curve around center
                    bandwidth = center / (q_factor * PHI)
                    bell = np.exp(-((np.abs(freqs) - center) / bandwidth) ** 2)
                    
                    # Apply gain
                    gain = 10 ** (gain_db / 20)
                    coeffs[ch] *= (1 + (gain - 1) * bell)
            
            return coeffs
        
        # Use overlap-add for real-time
        return rft_overlap_add(rft_eq_process, buffer, 
                               window_size=2048, hop_size=512)
    
    device = DeviceNode(
        kind=DeviceKind.EFFECT,
        subtype='rft_eq',
        name='Φ-RFT EQ',
        params={
            'base_freq': 440.0,
            'band_0_db': 0.0,  # PHI^-2 = 0.382x
            'band_1_db': 0.0,  # PHI^-1 = 0.618x  
            'band_2_db': 0.0,  # PHI^0 = 1x (base)
            'band_3_db': 0.0,  # PHI^1 = 1.618x
            'band_4_db': 0.0,  # PHI^2 = 2.618x
            'band_5_db': 0.0,  # PHI^3 = 4.236x
            'q': 1.0,
        },
        param_specs={
            'base_freq': {'min': 20, 'max': 10000, 'default': 440, 'unit': 'Hz'},
            'band_0_db': {'min': -24, 'max': 24, 'default': 0, 'unit': 'dB'},
            'band_1_db': {'min': -24, 'max': 24, 'default': 0, 'unit': 'dB'},
            'band_2_db': {'min': -24, 'max': 24, 'default': 0, 'unit': 'dB'},
            'band_3_db': {'min': -24, 'max': 24, 'default': 0, 'unit': 'dB'},
            'band_4_db': {'min': -24, 'max': 24, 'default': 0, 'unit': 'dB'},
            'band_5_db': {'min': -24, 'max': 24, 'default': 0, 'unit': 'dB'},
            'q': {'min': 0.1, 'max': 10, 'default': 1, 'unit': ''},
        },
        inputs=[DevicePort('audio_in', 'audio', 2)],
        outputs=[DevicePort('audio_out', 'audio', 2)]
    )
    device._processor = process
    device._state = {}
    return device


# ═══════════════════════════════════════════════════════════════════════════════
# RFT MORPH FILTER - Crossfade between spectral profiles
# ═══════════════════════════════════════════════════════════════════════════════

def create_rft_morph() -> DeviceNode:
    """
    RFT Morph Filter.
    Crossfades between two wave-domain "profiles" using Φ-weighted interpolation.
    """
    
    def process(buffer: np.ndarray, params: Dict, state: Dict,
                sample_rate: float, block_size: int) -> np.ndarray:
        
        def rft_morph_process(coeffs: np.ndarray) -> np.ndarray:
            channels, n = coeffs.shape
            
            morph = params.get('morph', 0.5)  # 0 = profile A, 1 = profile B
            resonance = params.get('resonance', 0.5)
            
            freqs = np.fft.fftfreq(n, 1/sample_rate)
            
            # Profile A: Low-pass character
            cutoff_a = params.get('cutoff_a', 1000.0)
            profile_a = 1.0 / (1.0 + (np.abs(freqs) / cutoff_a) ** 2)
            
            # Profile B: High-pass + resonant peak
            cutoff_b = params.get('cutoff_b', 5000.0)
            profile_b = (np.abs(freqs) / cutoff_b) ** 2 / (1.0 + (np.abs(freqs) / cutoff_b) ** 2)
            
            # Add resonance at Φ-related frequencies
            res_freq = cutoff_a * PHI
            res_peak = resonance * np.exp(-((np.abs(freqs) - res_freq) / (res_freq * 0.1)) ** 2)
            profile_a += res_peak
            
            # Φ-weighted morph (not linear!)
            phi_morph = morph ** PHI_INV  # Golden-ratio curve
            profile = (1 - phi_morph) * profile_a + phi_morph * profile_b
            
            for ch in range(channels):
                coeffs[ch] *= profile
            
            return coeffs
        
        return rft_overlap_add(rft_morph_process, buffer,
                               window_size=2048, hop_size=512)
    
    device = DeviceNode(
        kind=DeviceKind.EFFECT,
        subtype='rft_morph',
        name='Φ-Morph Filter',
        params={
            'morph': 0.5,
            'resonance': 0.3,
            'cutoff_a': 1000.0,
            'cutoff_b': 5000.0,
        },
        param_specs={
            'morph': {'min': 0, 'max': 1, 'default': 0.5, 'unit': ''},
            'resonance': {'min': 0, 'max': 1, 'default': 0.3, 'unit': ''},
            'cutoff_a': {'min': 20, 'max': 20000, 'default': 1000, 'unit': 'Hz'},
            'cutoff_b': {'min': 20, 'max': 20000, 'default': 5000, 'unit': 'Hz'},
        },
        inputs=[DevicePort('audio_in', 'audio', 2)],
        outputs=[DevicePort('audio_out', 'audio', 2)]
    )
    device._processor = process
    return device


# ═══════════════════════════════════════════════════════════════════════════════
# RFT FILTER - Basic LP/HP/BP with Φ-resonance
# ═══════════════════════════════════════════════════════════════════════════════

def create_rft_filter() -> DeviceNode:
    """
    RFT Filter with Φ-resonance.
    Low-pass, high-pass, band-pass modes with golden-ratio resonance peaks.
    """
    
    def process(buffer: np.ndarray, params: Dict, state: Dict,
                sample_rate: float, block_size: int) -> np.ndarray:
        
        def rft_filter_process(coeffs: np.ndarray) -> np.ndarray:
            channels, n = coeffs.shape
            
            cutoff = params.get('cutoff', 1000.0)
            resonance = params.get('resonance', 0.0)
            mode = int(params.get('mode', 0))  # 0=LP, 1=HP, 2=BP
            
            freqs = np.fft.fftfreq(n, 1/sample_rate)
            freq_ratio = np.abs(freqs) / max(cutoff, 1.0)
            
            if mode == 0:  # Low-pass
                response = 1.0 / (1.0 + freq_ratio ** 4)
            elif mode == 1:  # High-pass
                response = freq_ratio ** 4 / (1.0 + freq_ratio ** 4)
            else:  # Band-pass
                response = freq_ratio ** 2 / (1.0 + freq_ratio ** 4)
            
            # Add Φ-resonance at cutoff
            if resonance > 0:
                res_peak = resonance * np.exp(-((freq_ratio - 1) / 0.1) ** 2)
                response += res_peak
                
                # Secondary resonance at Φ × cutoff
                freq_ratio_phi = np.abs(freqs) / max(cutoff * PHI, 1.0)
                res_peak_phi = resonance * PHI_INV * np.exp(-((freq_ratio_phi - 1) / 0.1) ** 2)
                response += res_peak_phi
            
            for ch in range(channels):
                coeffs[ch] *= response
            
            return coeffs
        
        return rft_overlap_add(rft_filter_process, buffer,
                               window_size=1024, hop_size=256)
    
    device = DeviceNode(
        kind=DeviceKind.EFFECT,
        subtype='rft_filter',
        name='Φ-Filter',
        params={
            'cutoff': 1000.0,
            'resonance': 0.0,
            'mode': 0,
        },
        param_specs={
            'cutoff': {'min': 20, 'max': 20000, 'default': 1000, 'unit': 'Hz'},
            'resonance': {'min': 0, 'max': 1, 'default': 0, 'unit': ''},
            'mode': {'min': 0, 'max': 2, 'default': 0, 'unit': 'LP/HP/BP'},
        },
        inputs=[DevicePort('audio_in', 'audio', 2)],
        outputs=[DevicePort('audio_out', 'audio', 2)]
    )
    device._processor = process
    return device


# ═══════════════════════════════════════════════════════════════════════════════
# COMPRESSOR - Traditional dynamics with Φ-timing
# ═══════════════════════════════════════════════════════════════════════════════

def create_compressor() -> DeviceNode:
    """
    Compressor with Φ-based attack/release timing.
    """
    
    def process(buffer: np.ndarray, params: Dict, state: Dict,
                sample_rate: float, block_size: int) -> np.ndarray:
        
        threshold_db = params.get('threshold_db', -20.0)
        ratio = params.get('ratio', 4.0)
        attack_ms = params.get('attack_ms', 10.0)
        release_ms = params.get('release_ms', 100.0)
        makeup_db = params.get('makeup_db', 0.0)
        
        # Convert to linear
        threshold = 10 ** (threshold_db / 20)
        makeup = 10 ** (makeup_db / 20)
        
        # Time constants (Φ-scaled)
        attack_coef = np.exp(-1.0 / (attack_ms * sample_rate / 1000 * PHI))
        release_coef = np.exp(-1.0 / (release_ms * sample_rate / 1000 * PHI))
        
        # Get envelope follower state
        env = state.get('envelope', 0.0)
        
        output = np.zeros_like(buffer)
        
        for i in range(buffer.shape[1]):
            # Get input level (peak of both channels)
            input_level = np.max(np.abs(buffer[:, i]))
            
            # Envelope follower
            if input_level > env:
                env = attack_coef * env + (1 - attack_coef) * input_level
            else:
                env = release_coef * env + (1 - release_coef) * input_level
            
            # Compute gain reduction
            if env > threshold:
                gain_db = threshold_db + (20 * np.log10(max(env, 1e-10)) - threshold_db) / ratio
                gain = 10 ** (gain_db / 20) / max(env, 1e-10)
            else:
                gain = 1.0
            
            output[:, i] = buffer[:, i] * gain * makeup
        
        state['envelope'] = env
        return output
    
    device = DeviceNode(
        kind=DeviceKind.EFFECT,
        subtype='compressor',
        name='Compressor',
        params={
            'threshold_db': -20.0,
            'ratio': 4.0,
            'attack_ms': 10.0,
            'release_ms': 100.0,
            'makeup_db': 0.0,
        },
        param_specs={
            'threshold_db': {'min': -60, 'max': 0, 'default': -20, 'unit': 'dB'},
            'ratio': {'min': 1, 'max': 20, 'default': 4, 'unit': ':1'},
            'attack_ms': {'min': 0.1, 'max': 100, 'default': 10, 'unit': 'ms'},
            'release_ms': {'min': 10, 'max': 1000, 'default': 100, 'unit': 'ms'},
            'makeup_db': {'min': 0, 'max': 24, 'default': 0, 'unit': 'dB'},
        },
        inputs=[DevicePort('audio_in', 'audio', 2)],
        outputs=[DevicePort('audio_out', 'audio', 2)]
    )
    device._processor = process
    device._state = {'envelope': 0.0}
    return device


# ═══════════════════════════════════════════════════════════════════════════════
# RFT REVERB - Wave-domain convolution reverb
# ═══════════════════════════════════════════════════════════════════════════════

def create_rft_reverb() -> DeviceNode:
    """
    RFT Texture Reverb.
    Wave-domain feedback reverb with Φ-based decay structure.
    """
    
    def process(buffer: np.ndarray, params: Dict, state: Dict,
                sample_rate: float, block_size: int) -> np.ndarray:
        
        decay = params.get('decay', 0.5)
        size = params.get('size', 0.5)
        damping = params.get('damping', 0.5)
        mix = params.get('mix', 0.3)
        
        # Initialize state buffers if needed
        if 'delay_lines' not in state:
            # Φ-spaced delay times in samples
            delay_samples = [
                int(size * sample_rate * 0.01 * PHI ** i) 
                for i in range(6)
            ]
            state['delay_lines'] = [
                np.zeros((2, max(d, 1)), dtype=np.float32) 
                for d in delay_samples
            ]
            state['delay_pos'] = [0] * 6
        
        dry = buffer.copy()
        wet = np.zeros_like(buffer)
        
        for i in range(buffer.shape[1]):
            sample = buffer[:, i].copy()
            
            # Read from delay lines and sum
            reverb_sum = np.zeros(2, dtype=np.float32)
            for j, (dl, pos) in enumerate(zip(state['delay_lines'], state['delay_pos'])):
                if dl.shape[1] > 0:
                    reverb_sum += dl[:, pos] * (decay ** (j * PHI_INV))
            
            # Apply damping (simple lowpass)
            if 'prev_out' not in state:
                state['prev_out'] = np.zeros(2, dtype=np.float32)
            reverb_sum = (1 - damping) * reverb_sum + damping * state['prev_out']
            state['prev_out'] = reverb_sum.copy()
            
            # Write to delay lines
            feedback_sample = sample + reverb_sum * decay
            for j, (dl, pos) in enumerate(zip(state['delay_lines'], state['delay_pos'])):
                if dl.shape[1] > 0:
                    dl[:, pos] = feedback_sample
                    state['delay_pos'][j] = (pos + 1) % dl.shape[1]
            
            wet[:, i] = reverb_sum
        
        # Mix dry/wet
        output = (1 - mix) * dry + mix * wet
        return output
    
    device = DeviceNode(
        kind=DeviceKind.EFFECT,
        subtype='rft_reverb',
        name='Φ-Reverb',
        params={
            'decay': 0.5,
            'size': 0.5,
            'damping': 0.5,
            'mix': 0.3,
        },
        param_specs={
            'decay': {'min': 0, 'max': 0.99, 'default': 0.5, 'unit': ''},
            'size': {'min': 0.1, 'max': 2, 'default': 0.5, 'unit': ''},
            'damping': {'min': 0, 'max': 1, 'default': 0.5, 'unit': ''},
            'mix': {'min': 0, 'max': 1, 'default': 0.3, 'unit': ''},
        },
        inputs=[DevicePort('audio_in', 'audio', 2)],
        outputs=[DevicePort('audio_out', 'audio', 2)]
    )
    device._processor = process
    device._state = {}
    return device


# ═══════════════════════════════════════════════════════════════════════════════
# METER - Level metering
# ═══════════════════════════════════════════════════════════════════════════════

def create_meter() -> DeviceNode:
    """Level meter - passes audio through, exposes levels"""
    
    def process(buffer: np.ndarray, params: Dict, state: Dict,
                sample_rate: float, block_size: int) -> np.ndarray:
        
        # Calculate peak and RMS
        peak_l = np.max(np.abs(buffer[0])) if buffer.shape[0] > 0 else 0
        peak_r = np.max(np.abs(buffer[1])) if buffer.shape[0] > 1 else peak_l
        
        rms_l = np.sqrt(np.mean(buffer[0] ** 2)) if buffer.shape[0] > 0 else 0
        rms_r = np.sqrt(np.mean(buffer[1] ** 2)) if buffer.shape[0] > 1 else rms_l
        
        # Store in state for UI access
        state['peak_l'] = peak_l
        state['peak_r'] = peak_r
        state['rms_l'] = rms_l
        state['rms_r'] = rms_r
        state['peak_db_l'] = 20 * np.log10(max(peak_l, 1e-10))
        state['peak_db_r'] = 20 * np.log10(max(peak_r, 1e-10))
        
        return buffer  # Pass through unchanged
    
    device = DeviceNode(
        kind=DeviceKind.METER,
        subtype='meter',
        name='Meter',
        params={},
        inputs=[DevicePort('audio_in', 'audio', 2)],
        outputs=[DevicePort('audio_out', 'audio', 2)]
    )
    device._processor = process
    device._state = {
        'peak_l': 0, 'peak_r': 0,
        'rms_l': 0, 'rms_r': 0,
        'peak_db_l': -60, 'peak_db_r': -60
    }
    return device


# ═══════════════════════════════════════════════════════════════════════════════
# EXPORTS
# ═══════════════════════════════════════════════════════════════════════════════

__all__ = [
    'create_device',
    'create_utility',
    'create_rft_eq',
    'create_rft_filter',
    'create_rft_morph',
    'create_compressor',
    'create_rft_reverb',
    'create_meter',
]
