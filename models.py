"""
Quantonium OS - API Data Models

Defines Pydantic-style request schemas for symbolic endpoints and quantum API.
Includes comprehensive validation for all API inputs.
"""

from pydantic import BaseModel, Field, validator
from typing import List, Dict, Any, Optional, Union
import re
import base64

# Maximum length constraints for security
MAX_KEY_LENGTH = 256
MAX_PLAINTEXT_LENGTH = 8192  # 8KB
MAX_CIPHERTEXT_LENGTH = 1024
MAX_WAVEFORM_LENGTH = 64
MIN_WAVEFORM_LENGTH = 3

# Pattern validation for specific fields
HASH_PATTERN = r'^[A-Za-z0-9+/=]+$'  # Base64 pattern for container hashes

class EncryptRequest(BaseModel):
    """Request model for encrypting data using resonance techniques."""
    plaintext: str = Field(
        ..., 
        example="hello world",
        description="Text to encrypt"
    )
    key: constr(min_length=4, max_length=MAX_KEY_LENGTH) = Field(
        ..., 
        example="symbolic-key",
        description="Encryption key"
    )
    
    @validator('plaintext')
    def validate_plaintext(cls, v):
        if not v.strip():
            raise ValueError("Plaintext cannot be empty or whitespace")
        return v
    
    @validator('key')
    def validate_key(cls, v):
        if not v.strip():
            raise ValueError("Key cannot be empty or whitespace")
        return v

class RFTRequest(BaseModel):
    """Request model for Resonance Fourier Transform operations."""
    waveform: conlist(float, min_items=MIN_WAVEFORM_LENGTH, max_items=MAX_WAVEFORM_LENGTH) = Field(
        ..., 
        example=[0.1, 0.5, 0.9],
        description="Waveform data as a list of floating point values"
    )
    
    @validator('waveform')
    def validate_waveform_values(cls, v):
        # Ensure all values are in range [0.0, 1.0]
        for val in v:
            if val < 0.0 or val > 1.0:
                raise ValueError(f"Waveform values must be between 0.0 and 1.0, got {val}")
        return v

class EntropyRequest(BaseModel):
    """Request model for generating quantum-inspired entropy."""
    amount: int = Field(
        32, 
        ge=1, 
        le=1024, 
        example=64,
        description="Amount of entropy to generate in bytes"
    )

class DecryptRequest(BaseModel):
    """Request model for decrypting data using resonance techniques."""
    ciphertext: constr(min_length=1, max_length=MAX_CIPHERTEXT_LENGTH, regex=HASH_PATTERN) = Field(
        ..., 
        example="Pm0C/vQlcaH4OL71EYN5zlIcWyltZNnjlD6XQqZwTLvLSkwfu2xvUA==",
        description="Ciphertext to decrypt (base64 encoded)"
    )
    key: constr(min_length=4, max_length=MAX_KEY_LENGTH) = Field(
        ..., 
        example="symbolic-key",
        description="Decryption key"
    )
    
    @validator('ciphertext')
    def validate_ciphertext(cls, v):
        # Validate that it's properly base64 encoded
        try:
            # Just checking format, not actually using the decoded value
            base64.b64decode(v)
        except Exception:
            raise ValueError("Ciphertext must be valid base64 encoded data")
        return v
        
    @validator('key')
    def validate_key(cls, v):
        if not v.strip():
            raise ValueError("Key cannot be empty or whitespace")
        return v

class ContainerUnlockRequest(BaseModel):
    """Request model for unlocking symbolic containers."""
    waveform: conlist(float, min_items=MIN_WAVEFORM_LENGTH, max_items=MAX_WAVEFORM_LENGTH) = Field(
        ..., 
        example=[0.2, 0.7, 0.3],
        description="Waveform data as a list of floating point values"
    )
    hash: constr(min_length=1, max_length=MAX_CIPHERTEXT_LENGTH, regex=HASH_PATTERN) = Field(
        ..., 
        example="Pm0C/vQlcaH4OL71EYN5zlIcWyltZNnjlD6XQqZwTLvLSkwfu2xvUA==",
        description="Container hash identifier"
    )
    key: constr(min_length=4, max_length=MAX_KEY_LENGTH) = Field(
        ..., 
        example="symbolic-key",
        description="Decryption key"
    )
    
    @validator('waveform')
    def validate_waveform_values(cls, v):
        # Ensure all values are in range [0.0, 1.0]
        for val in v:
            if val < 0.0 or val > 1.0:
                raise ValueError(f"Waveform values must be between 0.0 and 1.0, got {val}")
        return v

class SignRequest(BaseModel):
    """Request model for signing data using wave HMAC."""
    message: constr(min_length=1, max_length=MAX_PLAINTEXT_LENGTH) = Field(
        ..., 
        example="message to sign",
        description="Message to sign"
    )
    key: constr(min_length=4, max_length=MAX_KEY_LENGTH) = Field(
        ..., 
        example="symbolic-key",
        description="Signing key"
    )
    use_phase_info: bool = Field(
        True, 
        example=True,
        description="Whether to include phase information in signature"
    )

class VerifyRequest(BaseModel):
    """Request model for verifying signatures using wave HMAC."""
    message: constr(min_length=1, max_length=MAX_PLAINTEXT_LENGTH) = Field(
        ..., 
        example="message to verify",
        description="Original message"
    )
    signature: constr(min_length=1, max_length=MAX_CIPHERTEXT_LENGTH, regex=HASH_PATTERN) = Field(
        ..., 
        example="Pm0C/vQlcaH4OL71EYN5zlIcWyltZNnjlD6XQqZwTLvLSkwfu2xvUA==",
        description="Signature to verify (base64 encoded)"
    )
    key: constr(min_length=4, max_length=MAX_KEY_LENGTH) = Field(
        ..., 
        example="symbolic-key",
        description="Verification key"
    )
    use_phase_info: bool = Field(
        True, 
        example=True,
        description="Whether phase information was used in signature"
    )

class IRFTRequest(BaseModel):
    """Request model for Inverse Resonance Fourier Transform operations."""
    frequency_data: Dict[str, Any] = Field(
        ..., 
        example={
            "frequencies": [0.1, 0.3, 0.5],
            "amplitudes": [0.7, 0.3, 0.1],
            "phases": [0.0, 0.25, 0.5]
        },
        description="Frequency domain data with frequencies, amplitudes, and phases"
    )
    
    @validator('frequency_data')
    def validate_frequency_data(cls, v):
        # Check required keys
        required_keys = ['frequencies', 'amplitudes', 'phases']
        for key in required_keys:
            if key not in v:
                raise ValueError(f"Missing required key '{key}' in frequency_data")
            
        # Check that all arrays have the same length
        lengths = [len(v[key]) for key in required_keys]
        if not all(length == lengths[0] for length in lengths):
            raise ValueError("All frequency data arrays must have the same length")
        
        # Check values are in proper range
        for amp in v['amplitudes']:
            if not 0.0 <= amp <= 1.0:
                raise ValueError(f"Amplitude values must be between 0.0 and 1.0, got {amp}")
                
        for phase in v['phases']:
            if not 0.0 <= phase <= 1.0:
                raise ValueError(f"Phase values must be between 0.0 and 1.0, got {phase}")
        
        return v

# Structured quantum circuit validation
class Gate(BaseModel):
    """Model for a quantum gate definition."""
    name: str = Field(..., example="h")
    target: int = Field(..., example=0, ge=0)
    control: Optional[int] = Field(None, example=1, ge=0)

class QuantumCircuit(BaseModel):
    """Model for a complete quantum circuit definition."""
    gates: List[Gate] = Field(..., example=[
        {"name": "h", "target": 0},
        {"name": "cnot", "control": 0, "target": 1}
    ])

class QuantumCircuitRequest(BaseModel):
    """Request model for quantum circuit processing API."""
    circuit: QuantumCircuit = Field(
        ..., 
        example={
            "gates": [
                {"name": "h", "target": 0},
                {"name": "cnot", "control": 0, "target": 1}
            ]
        },
        description="Quantum circuit definition"
    )
    qubit_count: int = Field(
        3, 
        ge=1, 
        le=150, 
        example=3,
        description="Number of qubits to use in the circuit (1-150)"
    )
    
    @validator('circuit')
    def validate_circuit_targets(cls, v, values):
        if 'qubit_count' in values:
            qubit_count = values['qubit_count']
            for gate in v.gates:
                if gate.target >= qubit_count:
                    raise ValueError(f"Gate target {gate.target} is out of range for qubit_count {qubit_count}")
                if gate.control is not None and gate.control >= qubit_count:
                    raise ValueError(f"Gate control {gate.control} is out of range for qubit_count {qubit_count}")
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
