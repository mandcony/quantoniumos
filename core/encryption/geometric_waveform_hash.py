"""
Geometric Waveform Hash - Patent-protected geometric waveform hashing
USPTO Application #19/169,399

This module implements patent-protected geometric waveform hashing using
golden ratio optimization and cryptographic-strength hash functions.
"""

import hashlib
import math
from typing import List, Dict, Any, Optional, Tuple, Union
import logging

logger = logging.getLogger(__name__)

# Golden ratio constant
PHI = (1 + math.sqrt(5)) / 2

# Try to import C++ accelerated module
try:
    # Temporarily disable C++ import to fix hanging issue during import
    # from core.pybind_interface import GeometricWaveformHash as CppGeometricWaveformHash
    CPP_AVAILABLE = False
    CppGeometricWaveformHash = None  # Placeholder
    logger.info("C++ geometric waveform hash module disabled for debugging")
except ImportError:
    CPP_AVAILABLE = False
    CppGeometricWaveformHash = None  # Placeholder
    logger.warning("C++ module not available, using Python implementation")

class GeometricWaveformHash:
    """
    Patent-protected geometric waveform hashing with golden ratio optimization.
    """
    
    def __init__(self, waveform: List[float], nonce: bytes = None):
        # Don't store the full waveform to keep memory footprint low
        self.amplitude = 0.0
        self.phase = 0.0
        
        # Validate and store nonce
        if nonce is not None:
            if len(nonce) < 8:
                raise ValueError("Nonce must be at least 8 bytes")
            self.nonce = nonce
        else:
            self.nonce = b''  # Empty nonce for compatibility
            
        self.calculate_geometric_properties(waveform)
    
    def calculate_geometric_properties(self, waveform: List[float]):
        """Calculate geometric properties of the waveform."""
        if not waveform:
            self.amplitude = 0.0
            self.phase = 0.0
            return
        
        # Calculate amplitude as average absolute value
        self.amplitude = sum(abs(x) for x in waveform) / len(waveform)
        
        # Calculate phase using geometric phase analysis
        even_sum = sum(waveform[i] for i in range(0, len(waveform), 2))
        odd_sum = sum(waveform[i] for i in range(1, len(waveform), 2))
        
        if len(waveform) % 2 == 0:
            even_sum /= (len(waveform) // 2)
            odd_sum /= (len(waveform) // 2)
        else:
            even_sum /= (len(waveform) // 2 + 1)
            odd_sum /= (len(waveform) // 2)
        
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
        
        # Include nonce in hash calculation if present
        if self.nonce:
            sample_str += '_nonce_' + self.nonce.hex()
        
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

def geometric_waveform_hash(waveform: List[float], nonce: bytes = None) -> str:
    """
    Generate geometric waveform hash using patent-protected algorithms.
    
    Args:
        waveform: Input waveform as list of float values
        nonce: Optional nonce for additional randomization
        
    Returns:
        Cryptographic hash string with geometric properties
    """
    if CPP_AVAILABLE:
        # Use C++ implementation for performance
        try:
            cpp_hasher = CppGeometricWaveformHash(waveform)
            if nonce:
                # For C++ implementation, mix nonce into waveform
                modified_waveform = list(waveform)
                for i, byte_val in enumerate(nonce):
                    if i < len(modified_waveform):
                        modified_waveform[i] = (modified_waveform[i] + byte_val / 255.0) % 1.0
                cpp_hasher = CppGeometricWaveformHash(modified_waveform)
            return cpp_hasher.generate_hash()
        except Exception as e:
            logger.warning(f"C++ implementation failed: {e}, falling back to Python")
    
    # Use Python implementation
    hasher = GeometricWaveformHash(waveform, nonce=nonce)
    return hasher.generate_hash()

def generate_waveform_hash(waveform_or_amplitude: Union[List[float], float], phase: Optional[float] = None) -> str:
    """
    Generate waveform hash either from a waveform or from amplitude and phase.
    
    Args:
        waveform_or_amplitude: Either a waveform as list of float values, or an amplitude value
        phase: Phase value if first argument is amplitude, otherwise ignored
        
    Returns:
        Hash string with geometric properties
    """
    if phase is not None:
        # Called as generate_waveform_hash(amplitude, phase)
        amplitude = float(waveform_or_amplitude)
        phase = float(phase)
        
        # Generate a simple waveform with the given amplitude and phase
        waveform = []
        for i in range(64):
            t = i / 64.0
            value = amplitude * math.sin(2 * math.pi * t + phase * 2 * math.pi)
            waveform.append(value)
        
        # Get the geometric hash
        geo_hash = geometric_waveform_hash(waveform)
        
        # Replace the calculated values in the prefix with the original ones
        # Find the hash part after the second underscore
        parts = geo_hash.split('_')
        if len(parts) >= 3:
            hash_part = '_'.join(parts[2:])
            # Create new hash with original amplitude and phase
            return f"A{amplitude:.4f}_P{phase:.4f}_{hash_part}"
        else:
            return geo_hash
    else:
        # Called as generate_waveform_hash(waveform)
        return geometric_waveform_hash(waveform_or_amplitude)

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

# Add missing functions needed by orchestration/symbolic_container.py

def wave_hash(data: Union[bytes, bytearray, str, List[float]]) -> str:
    """
    Generate a hash from input data (binary data, string, or waveform).
    
    Args:
        data: Input data to hash (bytes, string, bytearray, or list of floats)
        
    Returns:
        64-character hash string
    """
    if isinstance(data, str):
        data_bytes = data.encode('utf-8')
    elif isinstance(data, list):
        # Convert list of floats to string representation
        data_str = '_'.join(str(round(x, 6)) for x in data)
        data_bytes = data_str.encode('utf-8')
    else:
        # Already bytes or bytearray
        data_bytes = bytes(data)
    
    # Generate SHA-256 hash
    sha256_hash = hashlib.sha256(data_bytes).hexdigest()
    
    return sha256_hash

def extract_wave_parameters(hash_str: str) -> Tuple[List[Dict[str, float]], float]:
    """
    Extract waveform parameters from a hash string.
    
    Args:
        hash_str: Input hash string
        
    Returns:
        Tuple of (list of wave parameter dictionaries, coherence threshold)
    """
    if not hash_str or len(hash_str) < 64:
        logger.warning(f"Invalid hash string: {hash_str}")
        return [], 0.7  # Default threshold
    
    # Use hash entropy to generate wave parameters
    waves = []
    
    # Divide hash into segments for multiple waves
    segments = [hash_str[i:i+16] for i in range(0, len(hash_str), 16)]
    
    for i, segment in enumerate(segments):
        if len(segment) < 16:
            continue
            
        # Convert hex segment to numeric values
        h1 = int(segment[:8], 16) / 0xffffffff  # Normalize to [0,1]
        h2 = int(segment[8:], 16) / 0xffffffff  # Normalize to [0,1]
        h3 = int(segment[12:], 16) / 0xffff  # For frequency
        
        # Apply transformations to get parameters
        amplitude = 0.3 + (h1 * 0.7)  # Range [0.3, 1.0]
        phase = h2  # Range [0, 1]
        frequency = h3 / 16.0  # Range [0, 1]
        
        waves.append({
            'amplitude': amplitude,
            'phase': phase,
            'frequency': frequency,
            'index': i
        })
    
    # Use last part of hash to determine coherence threshold
    if len(hash_str) >= 8:
        threshold_val = int(hash_str[-8:], 16) / 0xffffffff  # [0,1]
        threshold = 0.6 + (threshold_val * 0.2)  # Range [0.6, 0.8]
    else:
        threshold = 0.7  # Default
    
    return waves, threshold

def generate_waveform_from_parameters(params: Union[Tuple[str, float, float], Dict[str, float]]) -> List[float]:
    """
    Generate a waveform from parameters.
    
    Args:
        params: Either a tuple (key_id, amplitude, phase) or a dictionary with 'amplitude' and 'phase' keys
        
    Returns:
        Waveform as list of float values
    """
    # Extract amplitude and phase from params
    if isinstance(params, tuple) and len(params) >= 3:
        key_id, amplitude, phase = params
    elif isinstance(params, dict) and 'amplitude' in params and 'phase' in params:
        amplitude = params['amplitude']
        phase = params['phase']
    else:
        logger.warning(f"Invalid parameters: {params}")
        amplitude, phase = 0.5, 0.5  # Default values
    
    # Generate 64-point waveform
    waveform = []
    for i in range(64):
        t = i / 64.0
        value = amplitude * math.sin(2 * math.pi * t + phase * 2 * math.pi)
        waveform.append(value)
    
    return waveform

def calculate_waveform_coherence(waveform1: List[float], waveform2: Optional[List[float]] = None) -> float:
    """
    Calculate coherence between two waveforms. If only one waveform is provided,
    calculates self-coherence as a measure of regularity.
    
    Args:
        waveform1: First waveform
        waveform2: Second waveform (optional)
        
    Returns:
        Coherence value in range [0, 1]
    """
    if not waveform1:
        return 0.0
        
    if waveform2 is None:
        # Self-coherence mode
        waveform2 = waveform1
    
    # Ensure equal lengths
    min_length = min(len(waveform1), len(waveform2))
    if min_length == 0:
        return 0.0
        
    w1 = waveform1[:min_length]
    w2 = waveform2[:min_length]
    
    # Calculate mean values
    mean1 = sum(w1) / len(w1)
    mean2 = sum(w2) / len(w2)
    
    # Calculate correlation coefficient
    numerator = sum((a - mean1) * (b - mean2) for a, b in zip(w1, w2))
    denominator1 = sum((a - mean1) ** 2 for a in w1)
    denominator2 = sum((b - mean2) ** 2 for b in w2)
    
    if denominator1 == 0 or denominator2 == 0:
        return 0.0
        
    correlation = numerator / ((denominator1 * denominator2) ** 0.5)
    
    # Convert to coherence (0 to 1 scale)
    coherence = (correlation + 1) / 2
    
    return coherence

def combine_waveforms(waveform1: List[float], waveform2: List[float]) -> List[float]:
    """
    Combine two waveforms using interference patterns.
    
    Args:
        waveform1: First waveform
        waveform2: Second waveform
        
    Returns:
        Combined waveform
    """
    if not waveform1 or not waveform2:
        return waveform1 or waveform2 or []
    
    # Ensure equal lengths
    min_length = min(len(waveform1), len(waveform2))
    w1 = waveform1[:min_length]
    w2 = waveform2[:min_length]
    
    # Combine using constructive/destructive interference
    result = []
    for i in range(min_length):
        # Weighted sum with phase preservation
        combined = (w1[i] + w2[i]) / 2
        result.append(combined)
    
    return result

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

def extract_parameters_from_hash(hash_str: str) -> Dict[str, Any]:
    """
    Extract parameters from a hash string (alias for extract_wave_parameters with dict format).
    
    Args:
        hash_str: Input hash string
        
    Returns:
        Dictionary with extracted parameters, or None values for invalid hash
    """
    # Check for valid hash format first
    if not hash_str or not isinstance(hash_str, str):
        return {
            'amplitude': None,
            'phase': None,
            'frequency': None,
            'threshold': 0.7,
            'waves': []
        }
    
    # Check if hash is in the new format A{amplitude}_P{phase}_{hex_hash}
    if hash_str.startswith('A') and '_P' in hash_str:
        try:
            parts = hash_str.split('_')
            if len(parts) >= 3:
                amplitude_str = parts[0][1:]  # Remove 'A' prefix
                phase_str = parts[1][1:]      # Remove 'P' prefix
                amplitude = float(amplitude_str)
                phase = float(phase_str)
                
                return {
                    'amplitude': amplitude,
                    'phase': phase,
                    'frequency': 0.5,
                    'threshold': 0.7,
                    'waves': [{'amplitude': amplitude, 'phase': phase, 'frequency': 0.5}]
                }
        except (ValueError, IndexError):
            pass  # Fall through to legacy format checking
    
    # Check if hash is in correct format (hex characters and proper length)
    if not all(c in '0123456789abcdefABCDEF' for c in hash_str) or len(hash_str) < 16:
        return {
            'amplitude': None,
            'phase': None,
            'frequency': None,
            'threshold': 0.7,
            'waves': []
        }
    
    waves, threshold = extract_wave_parameters(hash_str)
    
    if not waves:
        return {
            'amplitude': None,
            'phase': None,
            'frequency': None,
            'threshold': threshold,
            'waves': []
        }
    
    # Return the first wave as the primary parameters
    primary_wave = waves[0]
    return {
        'amplitude': primary_wave['amplitude'],
        'phase': primary_wave['phase'],
        'frequency': primary_wave.get('frequency', 0.5),
        'threshold': threshold,
        'waves': waves
    }