#!/usr/bin/env python3
"""
Python implementation for true_rft_engine_bindings using the true unitary transform

This module provides a Python implementation that mimics the interface of the 
C++ true_rft_engine_bindings module, using the TrueResonanceFourierTransform from
true_rft_exact.py as its core implementation.
"""

import os
import hashlib
import secrets
import numpy as np
import warnings
import sys
from pathlib import Path

# Add the proper path for importing
current_dir = Path(__file__).parent
if str(current_dir) not in sys.path:
    sys.path.append(str(current_dir))

# Import the true unitary transform
try:
    from true_rft_exact import TrueResonanceFourierTransform
    print("Successfully imported TrueResonanceFourierTransform")
except ImportError as e:
    print(f"Error importing TrueResonanceFourierTransform: {e}")
    warnings.warn(
        "Using fallback implementation. The true unitary transform is not available.",
        RuntimeWarning
    )

class TrueRFTEngine:
    """Python implementation of the TrueRFTEngine using the true unitary transform"""
    
    def __init__(self, size=16):
        """Initialize the TrueRFTEngine"""
        self.size = size
        self.initialized = False
        self.transformer = None
        print("Using TrueResonanceFourierTransform for TrueRFTEngine")
        
    def init(self):
        """Initialize the engine"""
        if not self.initialized:
            self.transformer = TrueResonanceFourierTransform(N=self.size)
            self.initialized = True
        return {"status": "SUCCESS", "engine": "TrueResonanceFourierTransform", "initialized": True}
        
    def compute_rft(self, data):
        """Compute the RFT of data using the true unitary transform"""
        if not self.initialized:
            self.init()
            
        if not isinstance(data, (bytes, bytearray, np.ndarray)):
            data = np.array(data, dtype=np.complex128)
        
        # Use the true unitary transform
        result = self.transformer.transform(data)
        return {"status": "SUCCESS", "result": result}
        
    def compute_inverse_rft(self, data):
        """Compute the inverse RFT of data using the true unitary transform"""
        if not self.initialized:
            self.init()
            
        if not isinstance(data, (bytes, bytearray, np.ndarray)):
            data = np.array(data, dtype=np.complex128)
            
        # Use the true unitary inverse transform
        result = self.transformer.inverse_transform(data)
        return {"status": "SUCCESS", "result": result}
        
    def generate_key(self, input_data=None, salt=None, length=32):
        """Generate a cryptographic key"""
        if not self.initialized:
            self.init()
            
        # Convert input to bytes if necessary
        if input_data is None:
            input_data = secrets.token_bytes(32)
        elif isinstance(input_data, str):
            input_data = input_data.encode()
            
        if salt is None:
            salt = secrets.token_bytes(16)
        elif isinstance(salt, str):
            salt = salt.encode()
            
        # Apply RFT to input data
        input_array = np.frombuffer(input_data, dtype=np.uint8).astype(np.complex128)
        if len(input_array) > self.size:
            input_array = input_array[:self.size]
        elif len(input_array) < self.size:
            padding = np.zeros(self.size - len(input_array), dtype=np.complex128)
            input_array = np.concatenate([input_array, padding])
            
        transformed = self.compute_rft(input_array)["result"]
        
        # Hash the transformed data with the salt
        hasher = hashlib.sha256()
        hasher.update(salt)
        hasher.update(np.real(transformed).tobytes())
        hasher.update(np.imag(transformed).tobytes())
        
        # Return the first 'length' bytes of the hash
        return hasher.digest()[:length]
        
    def encrypt(self, data, key):
        """Encrypt data using the key and RFT"""
        if not self.initialized:
            self.init()
            
        # Convert data to bytes if necessary
        if isinstance(data, str):
            data = data.encode()
            
        # Convert key to bytes if necessary
        if isinstance(key, str):
            key = key.encode()
            
        # Apply RFT to data
        data_array = np.frombuffer(data, dtype=np.uint8).astype(np.complex128)
        
        # Process in blocks of self.size
        blocks = []
        for i in range(0, len(data_array), self.size):
            block = data_array[i:i+self.size]
            if len(block) < self.size:
                padding = np.zeros(self.size - len(block), dtype=np.complex128)
                block = np.concatenate([block, padding])
                
            # Apply the true unitary transform
            transformed = self.transformer.transform(block)
            
            # XOR with key-derived values
            key_hash = hashlib.sha256(key + str(i).encode()).digest()
            key_array = np.frombuffer(key_hash, dtype=np.uint8).astype(np.complex128)
            if len(key_array) > self.size:
                key_array = key_array[:self.size]
            elif len(key_array) < self.size:
                key_padding = np.zeros(self.size - len(key_array), dtype=np.complex128)
                key_array = np.concatenate([key_array, key_padding])
                
            # Apply another transformation to mix key with data
            encrypted = transformed * key_array
            blocks.append(encrypted)
            
        # Concatenate all blocks
        result = np.concatenate(blocks)
        return result.tobytes()
        
    def decrypt(self, data, key):
        """Decrypt data using the key and inverse RFT"""
        if not self.initialized:
            self.init()
            
        # Convert key to bytes if necessary
        if isinstance(key, str):
            key = key.encode()
            
        # Convert data to complex array
        try:
            data_array = np.frombuffer(data, dtype=np.complex128)
        except:
            # If data is not already in complex format, treat as bytes
            data_array = np.frombuffer(data, dtype=np.uint8).astype(np.complex128)
            
        # Process in blocks of self.size
        blocks = []
        for i in range(0, len(data_array), self.size):
            block = data_array[i:i+self.size]
            if len(block) < self.size:
                padding = np.zeros(self.size - len(block), dtype=np.complex128)
                block = np.concatenate([block, padding])
                
            # Generate key-derived values
            key_hash = hashlib.sha256(key + str(i).encode()).digest()
            key_array = np.frombuffer(key_hash, dtype=np.uint8).astype(np.complex128)
            if len(key_array) > self.size:
                key_array = key_array[:self.size]
            elif len(key_array) < self.size:
                key_padding = np.zeros(self.size - len(key_array), dtype=np.complex128)
                key_array = np.concatenate([key_array, key_padding])
                
            # Reverse the key mixing
            decrypted_transform = block / key_array
            
            # Apply inverse transform
            decrypted = self.transformer.inverse_transform(decrypted_transform)
            blocks.append(decrypted)
            
        # Concatenate all blocks and convert to bytes
        result = np.concatenate(blocks)
        return result.real.astype(np.uint8).tobytes()

# For compatibility with C++ module interface
def engine_init():
    """Initialize the engine"""
    engine = TrueRFTEngine()
    engine.init()
    return 1  # Success

def engine_final():
    """Finalize the engine"""
    pass  # Nothing to do

# Create a global engine instance
_global_engine = None

def get_global_engine(size=16):
    """Get or create the global engine instance"""
    global _global_engine
    if _global_engine is None:
        _global_engine = TrueRFTEngine(size)
        _global_engine.init()
    return _global_engine
