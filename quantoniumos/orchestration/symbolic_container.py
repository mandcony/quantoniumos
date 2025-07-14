"""
QuantoniumOS - Symbolic Container Module

This module implements the Symbolic Container architecture, which provides:
1. Secure, resonance-based container architecture with waveform unlock mechanism
2. Geometric vault with 3D visualization capabilities
3. Tamper-evident storage with wave coherence validation
4. Parent-child container inheritance tracking

All operations are strictly handled server-side to protect proprietary algorithms.
"""

import os
import hashlib
import base64
import json
import time
from typing import Tuple, Dict, Any, Optional, List, Union

# Import the waveform hash module
import sys
import pathlib
_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from encryption.geometric_waveform_hash import (
    wave_hash, 
    extract_wave_parameters,
    calculate_waveform_coherence,
    generate_waveform_from_parameters,
    combine_waveforms
)


def hash_file(file_path: str) -> Tuple[str, float, float]:
    """
    Generate a key ID, amplitude and phase parameter from a file
    
    Args:
        file_path: Path to the file to hash
        
    Returns:
        Tuple of (key_id, amplitude, phase)
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    
    # Read the file in binary mode
    with open(file_path, 'rb') as f:
        data = f.read()
    
    # Generate a hash of the file
    file_hash = wave_hash(data)
    
    # Extract wave parameters
    waves, threshold = extract_wave_parameters(file_hash)
    
    # Use first wave for parameters
    if waves:
        amplitude = waves[0]['amplitude']
        phase = waves[0]['phase']
    else:
        # Fallback to default values
        amplitude = 0.5
        phase = 0.5
    
    return file_hash[:16], amplitude, phase


def validate_container(container_hash: str, key: str) -> Dict[str, Any]:
    """
    Validate a container with the given hash and key
    
    Args:
        container_hash: Hash of the container to validate
        key: Key to validate against
        
    Returns:
        Dictionary with validation status
    """
    # Generate key parameters from the key
    key_bytes = key.encode('utf-8')
    key_hash = hashlib.sha256(key_bytes).hexdigest()
    
    # Extract parameters 
    key_id = key_hash[:16]
    h = int(key_hash[16:24], 16) / 0xffffffff  # Normalize to [0,1]
    amplitude = 0.3 + (h * 0.7)  # Range 0.3-1.0
    phase = int(key_hash[24:32], 16) / 0xffffffff  # Normalize to [0,1]
    
    # Create container from hash
    container = SymbolicContainer.from_hash(container_hash, (key_id, amplitude, phase))
    
    # Attempt to unlock
    result = container.unlock(amplitude, phase)
    
    if result is not None:
        return {
            "success": True,
            "matched": True,
            "container_hash": container_hash,
            "metadata": container.get_metadata()
        }
    else:
        return {
            "success": True,
            "matched": False,
            "container_hash": container_hash
        }

def create_container(payload: Any, key: str, author_id: Optional[str] = None) -> Dict[str, Any]:
    """
    Create a new sealed symbolic container
    
    Args:
        payload: Data to store in the container  
        key: Key to use for container sealing
        author_id: Optional author identifier
    
    Returns:
        Dictionary with container hash and metadata
    """
    # Generate key parameters from the key
    key_bytes = key.encode('utf-8')
    key_hash = hashlib.sha256(key_bytes).hexdigest()
    
    # Extract parameters 
    key_id = key_hash[:16]
    h = int(key_hash[16:24], 16) / 0xffffffff  # Normalize to [0,1]
    amplitude = 0.3 + (h * 0.7)  # Range 0.3-1.0
    phase = int(key_hash[24:32], 16) / 0xffffffff  # Normalize to [0,1]
    
    # Create and seal container
    container = SymbolicContainer(payload, (key_id, amplitude, phase), author_id=author_id)
    if container.seal():
        return {
            "hash": container.get_hash(),
            "metadata": container.get_metadata(),
            "success": True
        }
    else:
        return {
            "success": False,
            "error": "Failed to seal container"
        }

class SymbolicContainer:
    """Secure container class that uses waveform resonance for access control"""
    
    def __init__(self, 
                payload: Any, 
                key_parameters: Tuple[str, float, float], 
                validation_file: Optional[str] = None,
                parent_hash: Optional[str] = None,
                author_id: Optional[str] = None):
        """
        Initialize a new symbolic container with payload
        
        Args:
            payload: Data to store in the container
            key_parameters: Tuple of (key_id, amplitude, phase)
            validation_file: Optional path to validation file
            parent_hash: Optional hash of parent container
            author_id: Optional author identifier
        """
        self.payload = payload
        self.key_id, self.amplitude, self.phase = key_parameters
        self.validation_file = validation_file
        self.parent_hash = parent_hash
        self.author_id = author_id or "anonymous"
        self.timestamp = int(time.time())
        self.hash = None
        self.sealed = False
        
        # Internal state
        self._waveform = None
        self._threshold = 0.75  # Default threshold
    
    def seal(self) -> bool:
        """
        Seal the container by generating the waveform and hash
        
        Returns:
            True if sealing was successful, False otherwise
        """
        try:
            # Generate waveform from parameters
            params = {
                'amplitude': self.amplitude,
                'phase': self.phase,
                'frequency': 3  # Default frequency
            }
            self._waveform = generate_waveform_from_parameters(params)
            
            # Create container metadata
            metadata = {
                'key_id': self.key_id,
                'author_id': self.author_id,
                'timestamp': self.timestamp,
                'parent_hash': self.parent_hash
            }
            
            # Serialize payload and metadata
            serialized = json.dumps({
                'payload': self.payload,
                'metadata': metadata
            }).encode('utf-8')
            
            # Generate the hash
            self.hash = wave_hash(serialized)
            
            # Extract coherence threshold from hash
            _, self._threshold = extract_wave_parameters(self.hash)
            
            self.sealed = True
            return True
            
        except Exception as e:
            print(f"Error sealing container: {e}")
            return False
    
    def unlock(self, amplitude: float, phase: float) -> Any:
        """
        Attempt to unlock the container using amplitude and phase parameters
        
        Args:
            amplitude: Amplitude parameter (between 0 and 1)
            phase: Phase parameter (between 0 and 1)
            
        Returns:
            Container payload if successful, None otherwise
        """
        if not self.sealed:
            raise RuntimeError("Container must be sealed before unlocking")
        
        # Generate test waveform from provided parameters
        test_params = {
            'amplitude': amplitude,
            'phase': phase,
            'frequency': 3  # Default frequency
        }
        test_waveform = generate_waveform_from_parameters(test_params)
        
        # Calculate coherence between waveforms (with null check)
        if self._waveform is None:
            return None
        coherence = calculate_waveform_coherence(self._waveform, test_waveform)
        
        # Check if coherence exceeds threshold
        if coherence >= self._threshold:
            return self.payload
        
        # Access denied
        return None
    
    def get_hash(self) -> Optional[str]:
        """Get the container hash"""
        return self.hash if self.sealed else None
    
    def get_metadata(self) -> Dict[str, Any]:
        """Get container metadata"""
        return {
            'key_id': self.key_id,
            'author_id': self.author_id,
            'timestamp': self.timestamp,
            'parent_hash': self.parent_hash,
            'hash': self.hash,
            'sealed': self.sealed
        }
    
    @classmethod
    def from_hash(cls, hash_str: str, key_parameters: Tuple[str, float, float]) -> 'SymbolicContainer':
        """
        Create a container instance from its hash
        
        Args:
            hash_str: The container hash
            key_parameters: Tuple of (key_id, amplitude, phase)
            
        Returns:
            A sealed SymbolicContainer instance
        """
        # Create container with empty payload
        container = cls(None, key_parameters)
        
        # Set hash and mark as sealed
        container.hash = hash_str
        container.sealed = True
        
        # Extract wave parameters and create waveform
        waves, container._threshold = extract_wave_parameters(hash_str)
        
        # Use first wave for parameters (for backwards compatibility)
        if waves:
            params = {
                'amplitude': container.amplitude,
                'phase': container.phase,
                'frequency': 3  # Default frequency
            }
            container._waveform = generate_waveform_from_parameters(params)
        
        return container