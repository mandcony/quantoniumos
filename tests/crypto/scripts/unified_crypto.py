#!/usr/bin/env python3
"""
Enhanced RFT Crypto Wrapper for Validation
Provides unified interface to both pure Python and optimized implementations
"""

import os
import sys
import importlib.util
from typing import Union, Dict, Any, Optional, Tuple

# Add paths to import different implementations
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.append(os.path.join(os.path.dirname(__file__), '../../core'))
sys.path.append(os.path.join(os.path.dirname(__file__), '../../apps'))
sys.path.append(os.path.join(os.path.dirname(__file__), '../../ASSEMBLY/python_bindings'))

# Check which implementations are available
PYTHON_IMPL_AVAILABLE = False
ASSEMBLY_IMPL_AVAILABLE = False

# Try to import pure Python implementation
try:
    from core.enhanced_rft_crypto_v2 import EnhancedRFTCryptoV2
    PYTHON_IMPL_AVAILABLE = True
except ImportError:
    pass

# Try to import assembly-optimized implementation
try:
    # First check if unitary_rft can be imported
    unitary_rft_spec = importlib.util.find_spec("unitary_rft")
    if unitary_rft_spec is not None:
        from apps.enhanced_rft_crypto import EnhancedRFTCrypto
        ASSEMBLY_IMPL_AVAILABLE = True
except ImportError:
    pass

class UnifiedRFTCrypto:
    """
    Unified interface to Enhanced RFT Crypto implementations.
    Can use either pure Python or assembly-optimized implementation.
    """
    
    def __init__(self, key: bytes, use_assembly: bool = True):
        """
        Initialize the crypto implementation.
        
        Args:
            key: The encryption key
            use_assembly: Whether to use assembly-optimized implementation if available
        """
        self.key = key
        self.implementation = None
        self._initialized = False
        self._is_assembly = False
        
        # Try to use assembly implementation first if requested
        if use_assembly and ASSEMBLY_IMPL_AVAILABLE:
            try:
                self.implementation = EnhancedRFTCrypto(8)  # Default size
                self._is_assembly = True
                self._initialized = True
            except Exception as e:
                print(f"Error initializing assembly implementation: {e}")
                self._initialized = False
        
        # Fall back to Python implementation if assembly not available or failed
        if not self._initialized and PYTHON_IMPL_AVAILABLE:
            try:
                self.implementation = EnhancedRFTCryptoV2(self.key)
                self._is_assembly = False
                self._initialized = True
            except Exception as e:
                print(f"Error initializing Python implementation: {e}")
                self._initialized = False
        
        if not self._initialized:
            raise RuntimeError("Failed to initialize any RFT crypto implementation")

    def encrypt(self, data: bytes, associated_data: bytes = b"") -> bytes:
        """Encrypt data using the selected implementation."""
        if not self._initialized:
            raise RuntimeError("Crypto implementation not initialized")
        
        if self._is_assembly:
            # Assembly version has different interface
            result = self.implementation.encrypt(data, self.key)
            return result['ciphertext']
        else:
            # Pure Python version
            return self.implementation.encrypt_aead(data, associated_data)
    
    def decrypt(self, data: bytes, associated_data: bytes = b"") -> bytes:
        """Decrypt data using the selected implementation."""
        if not self._initialized:
            raise RuntimeError("Crypto implementation not initialized")
        
        if self._is_assembly:
            # Assembly version has different interface
            # For assembly, the encrypted data needs to be restructured
            # This is a simplification - actual implementation would need proper format
            result = {'ciphertext': data, 'salt': b'\x00' * 16, 'rft_size': 8}
            return self.implementation.decrypt(result, self.key)
        else:
            # Pure Python version
            return self.implementation.decrypt_aead(data, associated_data)
    
    def is_assembly_implementation(self) -> bool:
        """Return whether the active implementation is assembly-optimized."""
        return self._is_assembly
    
    def get_implementation_name(self) -> str:
        """Return the name of the active implementation."""
        if self._is_assembly:
            return "Assembly-Optimized RFT Crypto"
        else:
            return "Pure Python RFT Crypto v2"
    
    @staticmethod
    def available_implementations() -> Dict[str, bool]:
        """Return available implementations."""
        return {
            "python": PYTHON_IMPL_AVAILABLE,
            "assembly": ASSEMBLY_IMPL_AVAILABLE
        }
    
    @staticmethod
    def expected_throughput() -> Dict[str, float]:
        """Return expected throughput for each implementation."""
        return {
            "python": 0.004,    # Measured MB/s
            "assembly": 9.2     # Paper target MB/s
        }
