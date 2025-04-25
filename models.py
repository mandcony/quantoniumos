"""
Quantonium OS - API Data Models

Defines Pydantic-style request schemas for symbolic endpoints and quantum API.
Enhanced with stronger validation and security constraints.
"""

import re
from pydantic import BaseModel, Field, validator
from typing import List, Dict, Any, Union, Optional

# Regular expressions for validation
HEX_PATTERN = r'^[0-9a-fA-F]+$'
BASE64_PATTERN = r'^[A-Za-z0-9+/]+={0,2}$'
HASH_PATTERN = r'^[0-9a-fA-F]{32,128}$'
KEY_PATTERN = r'^[A-Za-z0-9_\-]{8,64}$'

# Maximum sizes for security
MAX_PLAINTEXT_SIZE = 10485760  # 10 MB max payload
MAX_WAVEFORM_SIZE = 4096       # Maximum number of points in a waveform
MAX_KEY_SIZE = 1024            # Maximum key length in characters
MAX_CIRCUIT_DEPTH = 10000      # Maximum quantum circuit depth

class EncryptRequest(BaseModel):
    """Validated encryption request model with size limits."""
    plaintext: str = Field(
        ..., 
        example="hello world",
        description="Text to encrypt, max 10MB"
    )
    key: str = Field(
        ..., 
        example="symbolic-key123",
        description="Encryption key, alphanumeric with _ and - only"
    )
    
    @validator('plaintext')
    def validate_plaintext(cls, v):
        """Check if plaintext is valid"""
        if len(v) > MAX_PLAINTEXT_SIZE:
            raise ValueError(f'Plaintext too large (max {MAX_PLAINTEXT_SIZE} bytes)')
        return v
    
    @validator('key')
    def validate_key(cls, v):
        """Check if key meets complexity requirements"""
        if len(v) < 8 or len(v) > MAX_KEY_SIZE:
            raise ValueError(f'Key length must be between 8 and {MAX_KEY_SIZE} characters')
        if not re.match(KEY_PATTERN, v):
            raise ValueError('Key must contain only alphanumeric characters, underscores, and hyphens')
        if not any(c.isdigit() for c in v):
            raise ValueError('Key must contain at least one digit')
        if not any(c.isalpha() for c in v):
            raise ValueError('Key must contain at least one letter')
        return v

class RFTRequest(BaseModel):
    """Resonance Fourier Transform request with enhanced validation."""
    waveform: List[float] = Field(
        ..., 
        example=[0.1, 0.5, 0.9],
        description=f"Array of float values between 0 and 1, max {MAX_WAVEFORM_SIZE} points"
    )
    
    @validator('waveform')
    def validate_waveform_values(cls, v):
        """Check if waveform values are within valid range"""
        if len(v) < 2:
            raise ValueError(f'Waveform must have at least 2 points')
        if len(v) > MAX_WAVEFORM_SIZE:
            raise ValueError(f'Waveform too long (max {MAX_WAVEFORM_SIZE} points)')
        if not all(0 <= x <= 1 for x in v):
            raise ValueError('All waveform values must be between 0 and 1')
        return v

class EntropyRequest(BaseModel):
    """Secure entropy generation request."""
    amount: int = Field(
        32, 
        ge=1, 
        le=1024, 
        example=64,
        description="Amount of entropy to generate (1-1024 bytes)"
    )

class DecryptRequest(BaseModel):
    """Validated decryption request."""
    ciphertext: str = Field(
        ..., 
        example="d6a88f4f8e5d0a3b7c9e1e5f2a4b8c7d",
        description="Ciphertext hash to decrypt"
    )
    key: str = Field(
        ..., 
        example="symbolic-key123",
        description="Decryption key, alphanumeric with _ and - only"
    )
    
    @validator('ciphertext')
    def validate_ciphertext(cls, v):
        """Validate ciphertext is a valid hash"""
        if len(v) < 32 or len(v) > 1024:
            raise ValueError('Ciphertext hash must be between 32 and 1024 characters')
        if not re.match(HASH_PATTERN, v):
            raise ValueError('Ciphertext must be a valid hexadecimal hash')
        return v
    
    @validator('key')
    def validate_key(cls, v):
        """Check if key meets requirements"""
        if len(v) < 8 or len(v) > MAX_KEY_SIZE:
            raise ValueError(f'Key length must be between 8 and {MAX_KEY_SIZE} characters')
        if not re.match(KEY_PATTERN, v):
            raise ValueError('Key must contain only alphanumeric characters, underscores, and hyphens')
        if not any(c.isdigit() for c in v):
            raise ValueError('Key must contain at least one digit')
        if not any(c.isalpha() for c in v):
            raise ValueError('Key must contain at least one letter')
        return v

class ContainerUnlockRequest(BaseModel):
    """Secure container unlock request with thorough validation."""
    waveform: List[float] = Field(
        ..., 
        example=[0.2, 0.7, 0.3],
        description=f"Waveform for resonance matching, max {MAX_WAVEFORM_SIZE} points"
    )
    hash: str = Field(
        ..., 
        example="d6a88f4f8e5d0a3b7c9e1e5f2a4b8c7d",
        description="Container hash to unlock"
    )
    key: str = Field(
        ..., 
        example="symbolic-key123",
        description="Container key"
    )
    
    @validator('waveform')
    def validate_waveform_values(cls, v):
        """Check if waveform values are within valid range"""
        if len(v) < 2:
            raise ValueError(f'Waveform must have at least 2 points')
        if len(v) > MAX_WAVEFORM_SIZE:
            raise ValueError(f'Waveform too long (max {MAX_WAVEFORM_SIZE} points)')
        if not all(0 <= x <= 1 for x in v):
            raise ValueError('All waveform values must be between 0 and 1')
        return v
    
    @validator('hash')
    def validate_hash(cls, v):
        """Validate hash format"""
        if len(v) < 32 or len(v) > 128:
            raise ValueError('Container hash must be between 32 and 128 characters')
        if not re.match(HASH_PATTERN, v):
            raise ValueError('Container hash must be a valid hexadecimal string')
        return v
    
    @validator('key')
    def validate_key(cls, v):
        """Check if key meets requirements"""
        if len(v) < 8 or len(v) > MAX_KEY_SIZE:
            raise ValueError(f'Key length must be between 8 and {MAX_KEY_SIZE} characters')
        if not re.match(KEY_PATTERN, v):
            raise ValueError('Key must contain only alphanumeric characters, underscores, and hyphens')
        if not any(c.isdigit() for c in v):
            raise ValueError('Key must contain at least one digit')
        if not any(c.isalpha() for c in v):
            raise ValueError('Key must contain at least one letter')
        return v

class HMACSignRequest(BaseModel):
    """Request model for signing messages with wave_hmac."""
    message: str = Field(
        ...,
        example="message to sign",
        description="Message to sign"
    )
    key: str = Field(
        ...,
        example="secure-signing-key123",
        description="Signing key"
    )
    use_phase_info: bool = Field(
        True,
        example=True,
        description="Whether to include phase information in signature"
    )
    
    @validator('message')
    def validate_message(cls, v):
        """Validate message"""
        if len(v) < 1:
            raise ValueError('Message cannot be empty')
        if len(v) > MAX_PLAINTEXT_SIZE:
            raise ValueError(f'Message too large (max {MAX_PLAINTEXT_SIZE} bytes)')
        return v
    
    @validator('key')
    def validate_key(cls, v):
        """Check if key meets requirements"""
        if len(v) < 8 or len(v) > MAX_KEY_SIZE:
            raise ValueError(f'Key length must be between 8 and {MAX_KEY_SIZE} characters')
        if not re.match(KEY_PATTERN, v):
            raise ValueError('Key must contain only alphanumeric characters, underscores, and hyphens')
        if not any(c.isdigit() for c in v):
            raise ValueError('Key must contain at least one digit')
        if not any(c.isalpha() for c in v):
            raise ValueError('Key must contain at least one letter')
        return v

class HMACVerifyRequest(BaseModel):
    """Request model for verifying wave_hmac signatures."""
    message: str = Field(
        ...,
        example="message to verify",
        description="Original message"
    )
    signature: str = Field(
        ...,
        example="SGVsbG8gV29ybGQ=",
        description="Base64-encoded signature to verify"
    )
    key: str = Field(
        ...,
        example="secure-signing-key123",
        description="Verification key"
    )
    use_phase_info: bool = Field(
        True,
        example=True,
        description="Whether phase information was used in signature"
    )
    
    @validator('message')
    def validate_message(cls, v):
        """Validate message"""
        if len(v) < 1:
            raise ValueError('Message cannot be empty')
        if len(v) > MAX_PLAINTEXT_SIZE:
            raise ValueError(f'Message too large (max {MAX_PLAINTEXT_SIZE} bytes)')
        return v
    
    @validator('signature')
    def validate_signature(cls, v):
        """Validate signature format"""
        if len(v) < 16 or len(v) > 2048:
            raise ValueError('Signature must be between 16 and 2048 characters')
        if not re.match(BASE64_PATTERN, v):
            raise ValueError('Signature must be a valid base64-encoded string')
        return v
    
    @validator('key')
    def validate_key(cls, v):
        """Check if key meets requirements"""
        if len(v) < 8 or len(v) > MAX_KEY_SIZE:
            raise ValueError(f'Key length must be between 8 and {MAX_KEY_SIZE} characters')
        if not re.match(KEY_PATTERN, v):
            raise ValueError('Key must contain only alphanumeric characters, underscores, and hyphens')
        if not any(c.isdigit() for c in v):
            raise ValueError('Key must contain at least one digit')
        if not any(c.isalpha() for c in v):
            raise ValueError('Key must contain at least one letter')
        return v

class IRFTRequest(BaseModel):
    """Request model for Inverse Resonance Fourier Transform."""
    frequency_bins: List[float] = Field(
        ..., 
        example=[0.1, 0.5, 0.9, 0.5, 0.1],
        description="Array of frequency bin values"
    )
    harmonic_ratio: float = Field(
        ...,
        ge=0.0, 
        le=1.0,
        example=0.75,
        description="Harmonic ratio parameter (0-1)"
    )
    
    @validator('frequency_bins')
    def validate_frequency_bins(cls, v):
        """Validate frequency bins"""
        if len(v) < 2:
            raise ValueError('Frequency bins must have at least 2 points')
        if len(v) > MAX_WAVEFORM_SIZE:
            raise ValueError(f'Too many frequency bins (max {MAX_WAVEFORM_SIZE})')
        return v

class BenchmarkRequest(BaseModel):
    """Request model for the 64-perturbation benchmark test."""
    plaintext: Optional[str] = Field(
        None,
        example="deadbeef",
        description="Base plaintext in hex format (optional)"
    )
    key: Optional[str] = Field(
        None,
        example="c0ffee",
        description="Base key in hex format (optional)"
    )
    
    @validator('plaintext')
    def validate_plaintext(cls, v):
        """Validate plaintext is hex format"""
        if v is None:
            return v
        if len(v) < 2 or len(v) > 1024:
            raise ValueError('Plaintext must be between 2 and 1024 characters')
        if not re.match(HEX_PATTERN, v):
            raise ValueError('Plaintext must be a valid hexadecimal string')
        return v
    
    @validator('key')
    def validate_key(cls, v):
        """Validate key is hex format"""
        if v is None:
            return v
        if len(v) < 2 or len(v) > 1024:
            raise ValueError('Key must be between 2 and 1024 characters')
        if not re.match(HEX_PATTERN, v):
            raise ValueError('Key must be a valid hexadecimal string')
        return v

class QuantumCircuitRequest(BaseModel):
    """Request model for quantum circuit processing API with enhanced validation."""
    circuit: Dict[str, Any] = Field(
        ..., 
        example={
            "gates": [
                {"name": "h", "target": 0},
                {"name": "cnot", "control": 0, "target": 1}
            ]
        },
        description="Quantum circuit specification"
    )
    qubit_count: int = Field(
        3, 
        ge=1, 
        le=150, 
        example=3,
        description="Number of qubits in the circuit (1-150)"
    )
    
    @validator('circuit')
    def validate_circuit_structure(cls, v):
        """Validate circuit structure and gate definitions"""
        if 'gates' not in v:
            raise ValueError('Circuit must contain a "gates" array')
        
        gates = v.get('gates', [])
        if not isinstance(gates, list):
            raise ValueError('Gates must be an array')
            
        if len(gates) > MAX_CIRCUIT_DEPTH:
            raise ValueError(f'Circuit too deep (max {MAX_CIRCUIT_DEPTH} gates)')
            
        # Validate each gate
        valid_gate_types = {'h', 'x', 'y', 'z', 'cnot', 'swap', 'toffoli', 'measure'}
        for i, gate in enumerate(gates):
            if not isinstance(gate, dict):
                raise ValueError(f'Gate at position {i} is not an object')
                
            if 'name' not in gate:
                raise ValueError(f'Gate at position {i} missing "name" property')
                
            name = gate.get('name', '')
            if name not in valid_gate_types:
                raise ValueError(f'Gate "{name}" at position {i} is not a valid gate type')
                
            # Check target qubit
            if 'target' not in gate:
                raise ValueError(f'Gate at position {i} missing "target" property')
        
        return v
    
class QuantumBenchmarkRequest(BaseModel):
    """Request model for quantum benchmarking API."""
    max_qubits: int = Field(
        150, 
        ge=1, 
        le=150, 
        example=150,
        description="Maximum number of qubits to benchmark (1-150)"
    )
    run_full_benchmark: bool = Field(
        False, 
        example=True,
        description="Whether to run the full benchmark suite"
    )
