#!/usr/bin/env python3
# SPDX-License-Identifier: LicenseRef-QuantoniumOS-Claims-NC
# Copyright (C) 2025 Luis M. Minier / quantoniumos
"""
RFT Oscillator Implementation - QuantoniumOS
============================================

Oscillator based on Golden Ratio frequencies.
"""

import math
import numpy as np
from typing import Optional, Union, List

PHI = (1 + math.sqrt(5)) / 2

class Oscillator:
    """
    RFT Oscillator based on Golden Ratio frequencies.
    
    Frequencies are defined as: f_k = (k+1) * PHI
    """
    
    def __init__(self, mode: int = 0, phase: float = 0.0):
        """
        Initialize the oscillator.
        
        Args:
            mode: The RFT mode index (k). Sets frequency to (k+1)*PHI.
            phase: Initial phase in radians.
        """
        self.mode = mode
        self.phase = phase
        self.amplitude = 1.0
        self._update_frequency()
        
    def _update_frequency(self):
        self.frequency = (self.mode + 1) * PHI
        
    def set_mode(self, mode: int):
        """Set the RFT mode (k) and update frequency."""
        self.mode = mode
        self._update_frequency()

    def get_value(self, t: float) -> float:
        """
        Get the oscillator value at time t.
        
        y(t) = A * sin(2*pi*f*t + phi)
        """
        omega = 2 * math.pi * self.frequency
        return self.amplitude * math.sin(omega * t + self.phase)

    def encode_value(self, value: float):
        """
        Map data to RFT parameters.
        
        Encodes the value into the oscillator's amplitude.
        This supports BPSK (value = -1/1) or AM.
        
        Args:
            value: The data value to encode.
        """
        self.amplitude = value

    def decode_value(self, signal: np.ndarray, t: np.ndarray) -> float:
        """
        Decode value from a signal using correlation (matched filter).
        
        Args:
            signal: The signal array to decode from.
            t: The time array corresponding to the signal.
            
        Returns:
            The estimated amplitude/value.
        """
        # Generate reference carrier with unit amplitude
        omega = 2 * math.pi * self.frequency
        reference = np.sin(omega * t + self.phase)
        
        # Correlate: <signal, reference> / <reference, reference>
        # This projects the signal onto the carrier basis.
        numerator = np.dot(signal, reference)
        denominator = np.dot(reference, reference)
        
        if denominator == 0:
            return 0.0
            
        return numerator / denominator
