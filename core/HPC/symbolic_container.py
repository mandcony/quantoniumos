"""
Quantonium OS - Symbolic Container HPC Module

Implements the high-performance container validation routines for symbolic resonance containers.
"""

import hashlib
import sys
import os
import logging

# Try to import the HPC backend modules
try:
    # Add the bin directory to the path to find quantum_os.so
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../bin")))
    import quantum_os
    HPC_BACKEND_LOADED = True
    logger = logging.getLogger("symbolic_container")
    logger.info("✅ HPC quantum_os module loaded successfully")
except ImportError:
    # Fallback to basic implementation if HPC modules are not available
    HPC_BACKEND_LOADED = False
    logger = logging.getLogger("symbolic_container")
    logger.warning("⚠️ HPC quantum_os module not found, using fallback implementation")

def create_resonance_signature(waveform_array):
    """
    Create a resonance signature from the waveform array.
    If HPC backend is available, uses optimized C++ implementation.
    Otherwise, uses a Python fallback implementation.
    """
    if HPC_BACKEND_LOADED:
        try:
            # Call the C++ module if available
            return quantum_os.create_resonance_signature(waveform_array)
        except Exception as e:
            logger.error(f"Error in HPC create_resonance_signature: {str(e)}")
            # Fall through to fallback implementation
    
    # Fallback implementation (simulated resonance signature)
    if not waveform_array:
        return b""
    
    # Convert waveform to string and hash
    waveform_str = ",".join(str(x) for x in waveform_array)
    signature = hashlib.sha256(waveform_str.encode()).digest()[:8]
    
    return signature

def verify_container_waveform(signature, hash_value):
    """
    Verify a container using the resonance signature and hash.
    If HPC backend is available, uses optimized C++ implementation.
    Otherwise, uses a Python fallback implementation.
    """
    if HPC_BACKEND_LOADED:
        try:
            # Call the C++ module if available
            return quantum_os.verify_container_waveform(signature, hash_value)
        except Exception as e:
            logger.error(f"Error in HPC verify_container_waveform: {str(e)}")
            # Fall through to fallback implementation
    
    # Fallback implementation (simulated container verification)
    if not signature or not hash_value:
        return False
    
    # Simple hash check
    if isinstance(signature, bytes) and isinstance(hash_value, str):
        computed_hash = hashlib.sha256(signature).hexdigest()
        return computed_hash.startswith(hash_value)
    
    return False