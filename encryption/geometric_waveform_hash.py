"""
Geometric Waveform Hash - Patent-protected geometric waveform hashing
USPTO Application #19/169,399
"""

import hashlib
import math
from typing import List, Dict, Any, Optional
import logging
import os
import hmac

logger = logging.getLogger(__name__)

# Golden ratio constant
PHI = (1 + math.sqrt(5)) / 2

class GeometricWaveformHash:
    """
    Patent-protected geometric waveform hashing with golden ratio optimization.
    
    This implementation includes a nonce to salt the hash, enhancing security against
    predictability and collision attacks. The nonce should be a cryptographically
    secure random byte string.
    """
    
    def __init__(self, waveform: List[float], nonce: Optional[bytes] = None):
        """
        Initializes the hash generator.
        
        Args:
            waveform (List[float]): The input signal.
            nonce (Optional[bytes]): A random byte string used to salt the hash.
                                     It is highly recommended to provide a nonce of
                                     at least 16 bytes from a secure random source
                                     (e.g., os.urandom(16)).
        """
        self.waveform = waveform
        self.nonce = nonce
        self.amplitude = 0.0
        self.phase = 0.0
        self.calculate_geometric_properties()
    
    def calculate_geometric_properties(self):
        """Calculate geometric properties of the waveform."""
        if not self.waveform:
            self.amplitude = 0.0
            self.phase = 0.0
            return
        
        # Guard against single-value edge case
        if len(self.waveform) < 2:
            self.amplitude = 0.0
            self.phase = 0.0
            return
        
        # Calculate amplitude as average absolute value
        self.amplitude = sum(abs(x) for x in self.waveform) / len(self.waveform)
        
        # Calculate phase using geometric phase analysis
        if len(self.waveform) == 1:
            self.phase = 0.5  # Default phase for single value
        else:
            even_sum = sum(self.waveform[i] for i in range(0, len(self.waveform), 2))
            odd_sum = sum(self.waveform[i] for i in range(1, len(self.waveform), 2))
            
            even_count = (len(self.waveform) + 1) // 2
            odd_count = len(self.waveform) // 2
            
            if even_count > 0:
                even_sum /= even_count
            if odd_count > 0:
                odd_sum /= odd_count
            
            if even_sum == 0 and odd_sum == 0:
                self.phase = 0.0
            else:
                self.phase = math.atan2(odd_sum, even_sum) / (2 * math.pi)
                self.phase = (self.phase + 1.0) / 2.0  # Normalize to [0,1]
        
        # If a nonce is provided, use it to perturb the geometric properties
        # before applying the final golden ratio optimization. This salts the hash.
        if self.nonce:
            # Use the nonce to create a reproducible but unpredictable perturbation.
            # The nonce is converted to an integer and used to derive two salt values.
            if len(self.nonce) < 8:
                raise ValueError("Nonce must be at least 8 bytes long for effective salting.")
            
            nonce_int = int.from_bytes(self.nonce, 'big')
            
            # Derive two independent salt values from the nonce.
            salt1 = (nonce_int & 0xFFFFFFFF) / 0xFFFFFFFF  # Lower 32 bits
            salt2 = ((nonce_int >> 32) & 0xFFFFFFFF) / 0xFFFFFFFF  # Next 32 bits
            
            self.amplitude = (self.amplitude + salt1)
            self.phase = (self.phase + salt2)

        # Apply patent-protected golden ratio optimization
        self.amplitude = (self.amplitude * PHI) % 1.0
        self.phase = (self.phase * PHI) % 1.0
    
    def generate_hash(self) -> str:
        """
        Generate a cryptographic hash using an HMAC-SHA256 construction.
        
        The key is derived from the waveform's geometric properties (amplitude, phase)
        and the optional nonce. The message is the waveform data itself. This binds the
        waveform data to its geometric features, preventing collisions even if the
        geometric prefix is identical.
        """
        # The key for the HMAC is a combination of the geometric properties and the nonce.
        # This ensures that the hash is salted and tied to the waveform's unique geometry.
        nonce_hex = self.nonce.hex() if self.nonce else ""
        hmac_key = f"{self.amplitude:.17f}|{self.phase:.17f}|{nonce_hex}".encode('utf-8')
        
        # The message is the string representation of the raw waveform data.
        waveform_str = '_'.join(f"{x:.17f}" for x in self.waveform)
        hmac_message = waveform_str.encode('utf-8')
        
        # Compute the HMAC-SHA256 digest.
        hmac_digest = hmac.new(hmac_key, hmac_message, hashlib.sha256).hexdigest()
        
        # The final hash string includes a prefix of the geometric properties for
        # quick filtering, but the security is guaranteed by the HMAC digest.
        hash_str = f"A{self.amplitude:.5f}_P{self.phase:.5f}_{hmac_digest}"
        
        return hash_str
    
    def verify_hash(self, hash_str: str) -> bool:
        """Verify if the provided hash matches the current waveform."""
        return hmac.compare_digest(hash_str, self.generate_hash())
    
    def get_amplitude(self) -> float:
        """Get the calculated amplitude."""
        return self.amplitude
    
    def get_phase(self) -> float:
        """Get the calculated phase."""
        return self.phase

def geometric_waveform_hash(waveform: List[float], nonce: Optional[bytes] = None) -> str:
    """
    Convenience function to compute the geometric waveform hash.

    Args:
        waveform (List[float]): The input signal.
        nonce (Optional[bytes]): An optional nonce for salting. It is recommended
                                 to use os.urandom(16) to generate a secure nonce.

    Returns:
        str: The computed hash string.
    """
    hasher = GeometricWaveformHash(waveform, nonce)
    return hasher.generate_hash()

def generate_waveform_hash(waveform: List[float]) -> str:
    """Alias for geometric_waveform_hash for compatibility."""
    return geometric_waveform_hash(waveform)

def verify_waveform_hash(waveform: List[float], hash_str: str) -> bool:
    """Verify if the provided hash matches the waveform."""
    hasher = GeometricWaveformHash(waveform)
    return hasher.verify_hash(hash_str)

def get_waveform_properties(waveform: List[float]) -> Dict[str, float]:
    """Extract geometric properties from waveform."""
    hasher = GeometricWaveformHash(waveform)
    return {
        'amplitude': hasher.get_amplitude(),
        'phase': hasher.get_phase(),
        'golden_ratio': PHI,
        'waveform_length': len(waveform)
    }

def benchmark_geometric_hash(waveform_size: int = 32, iterations: int = 1000) -> Dict[str, Any]:
    """Benchmark geometric waveform hashing performance."""
    import time
    
    # Generate test waveform
    test_waveform = [math.sin(2 * math.pi * i / waveform_size) * 0.5 for i in range(waveform_size)]
    
    # Benchmark hashing
    start_time = time.time()
    for _ in range(iterations):
        geometric_waveform_hash(test_waveform)
    end_time = time.time()
    
    avg_time = (end_time - start_time) / iterations
    ops_per_second = 1.0 / avg_time if avg_time > 0 else 0
    
    return {
        'waveform_size': waveform_size,
        'iterations': iterations,
        'average_time_seconds': avg_time,
        'operations_per_second': ops_per_second,
        'golden_ratio': PHI
    }

# Legacy function aliases for compatibility
def wave_hash(waveform: List[float]) -> str:
    """Legacy alias for geometric_waveform_hash."""
    return geometric_waveform_hash(waveform)

def extract_wave_parameters(waveform: List[float]) -> Dict[str, float]:
    """Legacy alias for get_waveform_properties.""" 
    return get_waveform_properties(waveform)

def calculate_waveform_coherence(waveform: List[float]) -> float:
    """
    Calculate waveform coherence as a measure of signal quality.
    Returns a value between 0 and 1 indicating coherence level.
    """
    if len(waveform) < 2:
        return 0.0
    
    # Calculate amplitude variance as a coherence metric
    amplitudes = [abs(x) for x in waveform]
    mean_amp = sum(amplitudes) / len(amplitudes)
    
    if mean_amp == 0:
        return 0.0
    
    variance = sum((amp - mean_amp)**2 for amp in amplitudes) / len(amplitudes)
    coherence = 1.0 - min(variance / (mean_amp**2), 1.0)
    
    return coherence