#!/usr/bin/env python3
# SPDX-License-Identifier: LicenseRef-QuantoniumOS-Claims-NC
# Copyright (C) 2025 Luis M. Minier / quantoniumos
"""
Geometric Container Implementation - QuantoniumOS
=================================================

Geometric containers for resonant frequency encoding using RFT.
"""

import numpy as np
from typing import List, Dict, Optional
from .symbolic_wave_computer import SymbolicWaveComputer
from .oscillator import Oscillator

class LinearRegion:
    """
    Represents a linear region in the geometric space.
    """
    def __init__(self, start: float, end: float):
        self.start = start
        self.end = end

    def contains(self, value: float) -> bool:
        return self.start <= value <= self.end

    def __repr__(self):
        return f"LinearRegion({self.start}, {self.end})"

class GeometricContainer:
    """
    A container that holds data encoded as geometric waveforms using RFT principles.
    """
    def __init__(self, id: str, capacity_bits: int = 256):
        self.id = id
        self.swc = SymbolicWaveComputer(num_bits=capacity_bits)
        self.wave_form: Optional[np.ndarray] = None
        self.encoded_data_len: int = 0
        self.oscillators: List[Oscillator] = []
        self.resonant_frequencies: List[float] = []

    def encode_data(self, data: str) -> None:
        """
        Encodes string data into a wave form using SymbolicWaveComputer.
        
        Args:
            data: The string data to encode.
        """
        # Convert string to bytes
        byte_data = data.encode('utf-8')
        
        # Ensure SWC has enough capacity
        bits_needed = len(byte_data) * 8
        if bits_needed > self.swc.num_bits:
            # Resize SWC to accommodate the data
            self.swc = SymbolicWaveComputer(num_bits=bits_needed)
            
        self.wave_form = self.swc.encode(byte_data)
        self.encoded_data_len = len(byte_data)
        
        # Populate resonant frequencies based on active bits (simplified view)
        # In reality, SWC uses all carriers, but we can expose the carrier frequencies
        phi = (1 + np.sqrt(5)) / 2
        self.resonant_frequencies = [(k + 1) * phi for k in range(self.swc.num_bits)]

    def check_resonance(self, frequency: float, tolerance: float = 1e-3) -> bool:
        """
        Checks if a given frequency matches the RFT frequencies used in the encoded wave.
        
        Args:
            frequency: The frequency to check.
            tolerance: The tolerance for the frequency match.
            
        Returns:
            True if the frequency matches a carrier frequency, False otherwise.
        """
        # The RFT frequencies are f_k = (k + 1) * PHI
        phi = (1 + np.sqrt(5)) / 2
        
        # Inverse mapping: k = f / phi - 1
        # We check if the frequency corresponds to an integer index k
        k_approx = (frequency / phi) - 1
        k = int(round(k_approx))
        
        # Check if k is within the range of bits we are using
        if 0 <= k < self.swc.num_bits:
            expected_freq = (k + 1) * phi
            if abs(frequency - expected_freq) < tolerance:
                return True
                
        return False
        
    def get_data(self) -> Optional[str]:
        """Decode data from the waveform."""
        if self.wave_form is None:
            return None
        try:
            # Attempt to decode using SWC if available
            if hasattr(self.swc, 'decode_bytes'):
                decoded_bytes = self.swc.decode_bytes(self.wave_form)
                # Remove null padding (SWC pads with 0 bits -> null bytes)
                decoded_bytes = decoded_bytes.rstrip(b'\x00')
                return decoded_bytes.decode('utf-8')
            else:
                return "Data decoding not fully implemented in SWC wrapper"
        except Exception as e:
            return f"Error decoding data: {str(e)}"
