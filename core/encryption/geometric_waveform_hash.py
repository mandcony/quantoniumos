"""
Geometric Waveform Hash - Patent-protected geometric waveform hashing
USPTO Application #19/169,399

This module implements patent-protected geometric waveform hashing using
golden ratio optimization and cryptographic-strength hash functions.
"""

import hashlib
import math
from typing import List, Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)

# Golden ratio constant
PHI = (1 + math.sqrt(5)) / 2

# Try to import C++ accelerated module
try:
    from core.pybind_interface import GeometricWaveformHash as CppGeometricWaveformHash
    CPP_AVAILABLE = True
    logger.info("C++ geometric waveform hash module loaded successfully")
except ImportError:
    CPP_AVAILABLE = False
    logger.warning("C++ module not available, using Python implementation")

class GeometricWaveformHash:
    """
    Patent-protected geometric waveform hashing with golden ratio optimization.
    """
    
    def __init__(self, waveform: List[float]):
        self.waveform = waveform
        self.amplitude = 0.0
        self.phase = 0.0
        self.calculate_geometric_properties()
    
    def calculate_geometric_properties(self):
        """Calculate geometric properties of the waveform."""
        if not self.waveform:
            self.amplitude = 0.0
            self.phase = 0.0
            return
        
        # Calculate amplitude as average absolute value
        self.amplitude = sum(abs(x) for x in self.waveform) / len(self.waveform)
        
        # Calculate phase using geometric phase analysis
        even_sum = sum(self.waveform[i] for i in range(0, len(self.waveform), 2))
        odd_sum = sum(self.waveform[i] for i in range(1, len(self.waveform), 2))
        
        if len(self.waveform) % 2 == 0:
            even_sum /= (len(self.waveform) // 2)
            odd_sum /= (len(self.waveform) // 2)
        else:
            even_sum /= (len(self.waveform) // 2 + 1)
            odd_sum /= (len(self.waveform) // 2)
        
        self.phase = math.atan2(odd_sum, even_sum) / (2 * math.pi)
        self.phase = (self.phase + 1.0) / 2.0  # Normalize to [0,1]
        
        # Apply patent-protected golden ratio optimization
        self.amplitude = (self.amplitude * PHI) % 1.0
        self.phase = (self.phase * PHI) % 1.0
    
    def generate_hash(self) -> str:
        """Generate cryptographic hash of the geometric waveform."""
        # Generate 64-point waveform samples
        samples = []
        for i in range(64):
            t = i / 64.0
            value = self.amplitude * math.sin(2 * math.pi * t + self.phase * 2 * math.pi)
            samples.append(round(value, 6))  # 6 decimal precision
        
        # Create hash input string
        sample_str = '_'.join(str(s) for s in samples)
        
        # Generate SHA-256 hash
        sha256_hash = hashlib.sha256(sample_str.encode()).hexdigest()
        
        # Format final hash
        hash_str = f"A{self.amplitude:.4f}_P{self.phase:.4f}_{sha256_hash}"
        
        return hash_str
    
    def verify_hash(self, hash_str: str) -> bool:
        """Verify if the provided hash matches the current waveform."""
        return hash_str == self.generate_hash()
    
    def get_amplitude(self) -> float:
        """Get the calculated amplitude."""
        return self.amplitude
    
    def get_phase(self) -> float:
        """Get the calculated phase."""
        return self.phase

def geometric_waveform_hash(waveform: List[float]) -> str:
    """
    Generate geometric waveform hash using patent-protected algorithms.
    
    Args:
        waveform: Input waveform as list of float values
        
    Returns:
        Cryptographic hash string with geometric properties
    """
    if CPP_AVAILABLE:
        # Use C++ implementation for performance
        try:
            cpp_hasher = CppGeometricWaveformHash(waveform)
            return cpp_hasher.generate_hash()
        except Exception as e:
            logger.warning(f"C++ implementation failed: {e}, falling back to Python")
    
    # Use Python implementation
    hasher = GeometricWaveformHash(waveform)
    return hasher.generate_hash()

def generate_waveform_hash(waveform: List[float]) -> str:
    """Alias for geometric_waveform_hash for compatibility."""
    return geometric_waveform_hash(waveform)

def verify_waveform_hash(waveform: List[float], hash_str: str) -> bool:
    """
    Verify if the provided hash matches the waveform.
    
    Args:
        waveform: Input waveform as list of float values
        hash_str: Hash string to verify
        
    Returns:
        True if hash matches, False otherwise
    """
    if CPP_AVAILABLE:
        try:
            cpp_hasher = CppGeometricWaveformHash(waveform)
            return cpp_hasher.verify_hash(hash_str)
        except Exception as e:
            logger.warning(f"C++ verification failed: {e}, falling back to Python")
    
    # Use Python implementation
    hasher = GeometricWaveformHash(waveform)
    return hasher.verify_hash(hash_str)

def get_waveform_properties(waveform: List[float]) -> Dict[str, float]:
    """
    Extract geometric properties from waveform.
    
    Args:
        waveform: Input waveform as list of float values
        
    Returns:
        Dictionary with amplitude, phase, and other properties
    """
    if CPP_AVAILABLE:
        try:
            cpp_hasher = CppGeometricWaveformHash(waveform)
            return {
                'amplitude': cpp_hasher.get_amplitude(),
                'phase': cpp_hasher.get_phase(),
                'golden_ratio': PHI,
                'waveform_length': len(waveform)
            }
        except Exception as e:
            logger.warning(f"C++ properties extraction failed: {e}, falling back to Python")
    
    # Use Python implementation
    hasher = GeometricWaveformHash(waveform)
    return {
        'amplitude': hasher.get_amplitude(),
        'phase': hasher.get_phase(),
        'golden_ratio': PHI,
        'waveform_length': len(waveform)
    }

# Performance benchmark function
def benchmark_geometric_hash(waveform_size: int = 32, iterations: int = 1000) -> Dict[str, Any]:
    """
    Benchmark geometric waveform hashing performance.
    
    Args:
        waveform_size: Size of test waveform
        iterations: Number of iterations to run
        
    Returns:
        Performance metrics dictionary
    """
    import time
    
    # Generate test waveform
    test_waveform = [math.sin(2 * math.pi * i / waveform_size) * 0.5 
                    for i in range(waveform_size)]
    
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
        'cpp_available': CPP_AVAILABLE,
        'golden_ratio': PHI
    }