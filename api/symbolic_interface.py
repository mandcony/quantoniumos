"""
QuantoniumOS - Symbolic Interface

This module provides a secure interface to the proprietary QuantoniumOS core engine.
It uses ctypes/cffi to load and interface with the secure_core DLLs while maintaining
proper encapsulation of proprietary algorithms.
"""

import os
import sys
import ctypes
import logging
from ctypes import c_void_p, c_int, c_float, c_char_p, c_uint8, POINTER, Structure, Array
from typing import Dict, List, Any, Optional, Tuple
import base64
import hashlib

logger = logging.getLogger("quantonium_api.symbolic")

# Define C structures to match the DLL interface
class RFTResult(Structure):
    _fields_ = [
        ("bins", POINTER(c_float)),
        ("bin_count", c_int),
        ("hr", c_float)
    ]

class SAVector(Structure):
    _fields_ = [
        ("values", POINTER(c_float)),
        ("count", c_int)
    ]

# Global engine instance
_engine = None

def get_symbolic_engine():
    """
    Load and return the symbolic engine from secure_core
    Uses lazy initialization pattern
    """
    global _engine
    
    if _engine is not None:
        return _engine
    
    try:
        # Determine the appropriate extension for the current platform
        if sys.platform.startswith('win'):
            lib_extension = 'dll'
        elif sys.platform.startswith('darwin'):
            lib_extension = 'dylib'
        else:
            lib_extension = 'so'
            
        # Try to load the secure core library
        lib_path = os.path.abspath(f"secure_core/engine_core.{lib_extension}")
        logger.info(f"Loading symbolic engine from: {lib_path}")
        
        # Load the library
        lib = ctypes.cdll.LoadLibrary(lib_path)
        
        # Set up function prototypes
        
        # RFT calculation
        lib.rft_run.argtypes = [c_char_p, c_int]
        lib.rft_run.restype = POINTER(RFTResult)
        
        # RFT cleanup
        lib.rft_free.argtypes = [POINTER(RFTResult)]
        lib.rft_free.restype = None
        
        # Symbolic Alignment vector calculation
        lib.sa_compute.argtypes = [c_char_p, c_int]
        lib.sa_compute.restype = POINTER(SAVector)
        
        # SA cleanup
        lib.sa_free.argtypes = [POINTER(SAVector)]
        lib.sa_free.restype = None
        
        # Waveform hashing
        lib.wave_hash.argtypes = [c_char_p, c_int]
        lib.wave_hash.restype = c_char_p
        
        # Engine init/final
        lib.engine_init.argtypes = []
        lib.engine_init.restype = c_int
        
        lib.engine_final.argtypes = []
        lib.engine_final.restype = None
        
        # Initialize the engine
        status = lib.engine_init()
        if status != 0:
            logger.error(f"Failed to initialize symbolic engine: {status}")
            raise RuntimeError(f"Engine initialization failed with status: {status}")
            
        _engine = lib
        logger.info("Symbolic engine successfully loaded and initialized")
        return _engine
        
    except Exception as e:
        logger.error(f"Failed to load symbolic engine: {str(e)}")
        # In production, we don't want to reveal the exact error
        # But for development, it's useful to know what went wrong
        if os.environ.get("FLASK_ENV") == "development":
            raise
        else:
            raise RuntimeError("Failed to load secure symbolic engine")

def compute_rft_with_engine(data: str) -> Dict[str, Any]:
    """
    Compute Resonance Fourier Transform using the secure engine
    
    Args:
        data: The input data to analyze
        
    Returns:
        Dictionary with RFT bins and HR value
    """
    engine = get_symbolic_engine()
    
    # Prepare input data
    input_bytes = data.encode('utf-8')
    
    # Call RFT function
    result_ptr = engine.rft_run(input_bytes, len(input_bytes))
    
    # Copy results to avoid memory issues after freeing
    bins = []
    bin_count = result_ptr.contents.bin_count
    hr_value = result_ptr.contents.hr
    
    # Extract bin values
    for i in range(bin_count):
        bins.append(result_ptr.contents.bins[i])
    
    # Clean up memory
    engine.rft_free(result_ptr)
    
    return {
        "rft": bins,
        "hr": hr_value,
        "bin_count": bin_count
    }

def compute_sa_with_engine(data: str) -> Dict[str, Any]:
    """
    Compute Symbolic Alignment vector using the secure engine
    
    Args:
        data: The input data to analyze
        
    Returns:
        Dictionary with SA vector encoded as base64
    """
    engine = get_symbolic_engine()
    
    # Prepare input data
    input_bytes = data.encode('utf-8')
    
    # Call SA function
    result_ptr = engine.sa_compute(input_bytes, len(input_bytes))
    
    # Copy results to avoid memory issues after freeing
    sa_values = []
    count = result_ptr.contents.count
    
    # Extract values
    for i in range(count):
        sa_values.append(result_ptr.contents.values[i])
    
    # Clean up memory
    engine.sa_free(result_ptr)
    
    # Encode as base64 for transport
    # Convert floats to bytes first
    sa_bytes = b''
    for val in sa_values:
        sa_bytes += val.to_bytes(4, byteorder='little')
    
    sa_b64 = base64.b64encode(sa_bytes).decode('ascii')
    
    return {
        "sa_vector": sa_b64,
        "sa_count": count
    }

def compute_wave_hash(data: str) -> str:
    """
    Compute waveform hash using the secure engine
    
    Args:
        data: The input data to hash
        
    Returns:
        Hexadecimal hash string
    """
    engine = get_symbolic_engine()
    
    # Prepare input data
    input_bytes = data.encode('utf-8')
    
    # Call hash function
    hash_ptr = engine.wave_hash(input_bytes, len(input_bytes))
    
    # Copy hash string
    hash_str = ctypes.string_at(hash_ptr).decode('ascii')
    
    # No need to free the hash string as it's managed by the engine
    
    return hash_str

def safe_fallback_hash(data: str) -> str:
    """
    Fallback hash function in case the engine is not available
    This is NOT a complete implementation of the proprietary algorithm
    and should only be used for testing
    """
    logger.warning("Using fallback hash function - NOT SUITABLE FOR PRODUCTION")
    
    # Apply a custom serialization step to simulate waveform characteristics
    # This is NOT the actual proprietary algorithm
    serialized = ""
    for i, char in enumerate(data):
        serialized += chr((ord(char) + i) % 256)
    
    # Create SHA-256 hash
    sha256 = hashlib.sha256(serialized.encode()).hexdigest()
    
    return sha256