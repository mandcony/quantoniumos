"""
Quantonium OS - API Data Models

Defines Pydantic-style request schemas for symbolic endpoints and quantum API.
Includes strong validation for all API inputs.
"""

from pydantic import BaseModel, Field, validator
from typing import List, Dict, Any, Optional, Union
import re
import base64

# Enhanced base validation for keys and inputs
class EncryptRequest(BaseModel):
    """
    Request model for encryption API.
    Validates plaintext and key to ensure they meet security requirements.
    """
    plaintext: str = Field(
        ..., 
        description="Text to encrypt",
        example="hello world"
    )
    key: str = Field(
        ..., 
        description="Encryption key - minimum 8 characters",
        example="symbolic-key"
    )
    
    @validator('plaintext')
    def validate_plaintext_length(cls, v):
        """Validate plaintext is not too long or empty."""
        if not v or len(v) < 1:
            raise ValueError('Plaintext must not be empty')
        if len(v) > 10000:
            raise ValueError('Plaintext too long, maximum 10000 characters')
        return v
    
    @validator('key')
    def key_complexity(cls, v):
        """Validate key has sufficient complexity."""
        if len(v) < 8:
            raise ValueError('Key must be at least 8 characters')
        if len(v) > 128:
            raise ValueError('Key too long, maximum 128 characters')
        # Check for at least one number and one special char for stronger keys
        if not any(c.isdigit() for c in v) and not any(not c.isalnum() for c in v):
            raise ValueError('Key should contain at least one digit or special character for better security')
        return v

class RFTRequest(BaseModel):
    """
    Request model for Resonance Fourier Transform API.
    Validates waveform data to ensure it's properly formatted.
    """
    waveform: List[float] = Field(
        ..., 
        description="Waveform data as array of float values between 0 and 1",
        example=[0.1, 0.5, 0.9]
    )
    
    @validator('waveform')
    def validate_waveform_values(cls, v):
        """Ensure all waveform values are between 0 and 1 and list is properly sized."""
        if len(v) < 2:
            raise ValueError('Waveform must have at least 2 points')
        if len(v) > 2048:
            raise ValueError('Waveform too large, maximum 2048 points')
        if not all(0 <= x <= 1 for x in v):
            raise ValueError('All waveform values must be between 0 and 1')
        return v

class EntropyRequest(BaseModel):
    """
    Request model for entropy generation API.
    Validates that requested entropy amount is within allowed limits.
    """
    amount: int = Field(
        32, 
        ge=1, 
        le=1024, 
        description="Amount of entropy bytes to generate (1-1024)",
        example=64
    )

class DecryptRequest(BaseModel):
    """
    Request model for decryption API.
    Validates ciphertext format and key requirements.
    """
    ciphertext: str = Field(
        ..., 
        description="Ciphertext to decrypt (base64 encoded string)",
        example="base64_encoded_ciphertext_string"
    )
    key: str = Field(
        ..., 
        description="Decryption key - must match the key used for encryption",
        example="symbolic-key"
    )
    
    @validator('ciphertext')
    def validate_ciphertext_format(cls, v):
        """Ensure ciphertext has valid format."""
        if len(v) < 16:
            raise ValueError('Ciphertext too short, minimum 16 characters')
        if len(v) > 1024:
            raise ValueError('Ciphertext too long, maximum 1024 characters')
        # Check if it looks like a base64 string or hash value
        if not re.match(r'^[A-Za-z0-9+/=]+$', v):
            raise ValueError('Ciphertext must be a valid base64-encoded string')
        return v
    
    @validator('key')
    def key_complexity(cls, v):
        """Validate key has sufficient complexity."""
        if len(v) < 8:
            raise ValueError('Key must be at least 8 characters')
        if len(v) > 128:
            raise ValueError('Key too long, maximum 128 characters')
        return v

class ContainerUnlockRequest(BaseModel):
    """
    Request model for container unlock API.
    Validates waveform data, hash format, and key requirements.
    """
    waveform: List[float] = Field(
        ..., 
        description="Waveform data as array of float values between 0 and 1",
        example=[0.2, 0.7, 0.3]
    )
    hash: str = Field(
        ..., 
        description="Container hash identifier",
        example="d6a88f4f..."
    )
    key: str = Field(
        ..., 
        description="Encryption key used to create the container",
        example="symbolic-key"
    )
    
    @validator('waveform')
    def validate_waveform_values(cls, v):
        """Ensure all waveform values are between 0 and 1 and list is properly sized."""
        if len(v) < 2:
            raise ValueError('Waveform must have at least 2 points')
        if len(v) > 2048:
            raise ValueError('Waveform too large, maximum 2048 points')
        if not all(0 <= x <= 1 for x in v):
            raise ValueError('All waveform values must be between 0 and 1')
        return v
        
    @validator('hash')
    def validate_hash_format(cls, v):
        """Ensure hash has valid format."""
        if len(v) < 16:
            raise ValueError('Hash too short, minimum 16 characters')
        if len(v) > 1024:
            raise ValueError('Hash too long, maximum 1024 characters')
        # Check if it looks like a hash value
        if not re.match(r'^[A-Za-z0-9+/=]+$', v):
            raise ValueError('Hash must contain only alphanumeric characters, plus, slash, or equals')
        return v
    
    @validator('key')
    def validate_key(cls, v):
        """Validate key format."""
        if len(v) < 8:
            raise ValueError('Key must be at least 8 characters')
        if len(v) > 128:
            raise ValueError('Key too long, maximum 128 characters')
        return v

# Models for previously unvalidated endpoints
class SignRequest(BaseModel):
    """
    Request model for signature creation API.
    Validates message format and key requirements.
    """
    message: str = Field(
        ..., 
        description="Message to sign",
        example="Data to be signed for authenticity"
    )
    key: str = Field(
        ..., 
        description="Signing key",
        example="signing-key-123"
    )
    phase_info: bool = Field(
        True, 
        description="Whether to include phase information in the signature",
        example=True
    )
    
    @validator('message')
    def validate_message_length(cls, v):
        """Validate message is not too long or empty."""
        if not v or len(v) < 1:
            raise ValueError('Message must not be empty')
        if len(v) > 10000:
            raise ValueError('Message too long, maximum 10000 characters')
        return v
    
    @validator('key')
    def validate_key(cls, v):
        """Validate key format."""
        if len(v) < 8:
            raise ValueError('Key must be at least 8 characters')
        if len(v) > 128:
            raise ValueError('Key too long, maximum 128 characters')
        return v

class VerifyRequest(BaseModel):
    """
    Request model for signature verification API.
    Validates message, signature format, and key requirements.
    """
    message: str = Field(
        ..., 
        description="Original signed message",
        example="Data to be verified"
    )
    signature: str = Field(
        ..., 
        description="Signature to verify (base64 encoded)",
        example="base64_encoded_signature_string"
    )
    key: str = Field(
        ..., 
        description="Key used for signing",
        example="signing-key-123"
    )
    phase_info: bool = Field(
        True, 
        description="Whether phase information was used in the signature",
        example=True
    )
    
    @validator('message')
    def validate_message_length(cls, v):
        """Validate message is not too long or empty."""
        if not v or len(v) < 1:
            raise ValueError('Message must not be empty')
        if len(v) > 10000:
            raise ValueError('Message too long, maximum 10000 characters')
        return v
    
    @validator('signature')
    def validate_signature_format(cls, v):
        """Ensure signature has valid format."""
        if len(v) < 16:
            raise ValueError('Signature too short, minimum 16 characters')
        if len(v) > 1024:
            raise ValueError('Signature too long, maximum 1024 characters')
        # Check if it looks like a base64 string
        if not re.match(r'^[A-Za-z0-9+/=]+$', v):
            raise ValueError('Signature must be a valid base64-encoded string')
        return v
    
    @validator('key')
    def validate_key(cls, v):
        """Validate key format."""
        if len(v) < 8:
            raise ValueError('Key must be at least 8 characters')
        if len(v) > 128:
            raise ValueError('Key too long, maximum 128 characters')
        return v

class InverseRFTRequest(BaseModel):
    """
    Request model for Inverse Resonance Fourier Transform API.
    Validates frequency data format.
    """
    frequencies: List[Dict[str, Any]] = Field(
        ..., 
        description="Frequency components from RFT operation",
        example=[
            {"frequency": 1, "amplitude": 0.5, "phase": 0.25},
            {"frequency": 2, "amplitude": 0.3, "phase": 0.1}
        ]
    )
    output_length: int = Field(
        64, 
        ge=4, 
        le=2048, 
        description="Length of the output waveform",
        example=64
    )
    
    @validator('frequencies')
    def validate_frequency_components(cls, v):
        """Validate frequency component structure."""
        if not v or len(v) < 1:
            raise ValueError('At least one frequency component is required')
        
        for comp in v:
            if not all(key in comp for key in ['frequency', 'amplitude', 'phase']):
                raise ValueError('Each frequency component must have frequency, amplitude, and phase')
            
            # Validate amplitude and phase values are in appropriate ranges
            if not 0 <= comp.get('amplitude', 0) <= 1:
                raise ValueError('Amplitude must be between 0 and 1')
            
            if not 0 <= comp.get('phase', 0) < 2:  # Phase is typically 0 to 2π
                raise ValueError('Phase must be between 0 and 2')
                
        return v

class CircuitGate(BaseModel):
    """Model for a quantum circuit gate."""
    name: str = Field(..., description="Gate name (h, x, y, z, cnot, etc.)")
    target: int = Field(..., ge=0, description="Target qubit index")
    control: Optional[int] = Field(None, description="Control qubit index for controlled gates")
    
    @validator('name')
    def validate_gate_name(cls, v):
        """Validate gate name is from supported set."""
        valid_gates = {'h', 'x', 'y', 'z', 'cnot', 's', 't', 'rx', 'ry', 'rz', 'swap', 'cz', 'cp'}
        if v.lower() not in valid_gates:
            raise ValueError(f'Gate name must be one of: {", ".join(valid_gates)}')
        return v.lower()

class QuantumCircuit(BaseModel):
    """Model for a quantum circuit definition."""
    gates: List[CircuitGate] = Field(..., description="List of gates in the circuit")

class QuantumCircuitRequest(BaseModel):
    """
    Request model for quantum circuit processing API.
    Validates circuit specification and qubit count.
    """
    circuit: QuantumCircuit = Field(
        ..., 
        description="Quantum circuit definition",
        example={
            "gates": [
                {"name": "h", "target": 0},
                {"name": "cnot", "control": 0, "target": 1}
            ]
        }
    )
    qubit_count: int = Field(
        3, 
        ge=1, 
        le=150, 
        description="Number of qubits in the circuit (1-150)",
        example=3
    )
    
    @validator('circuit')
    def validate_circuit_gates(cls, v, values):
        """Validate gates reference valid qubits."""
        if 'qubit_count' not in values:
            return v
            
        qubit_count = values['qubit_count']
        for gate in v.gates:
            if gate.target >= qubit_count:
                raise ValueError(f'Gate target {gate.target} exceeds qubit count {qubit_count}')
            
            if gate.control is not None and gate.control >= qubit_count:
                raise ValueError(f'Gate control {gate.control} exceeds qubit count {qubit_count}')
                
            if gate.name.lower() in {'cnot', 'cz', 'cp', 'swap'} and gate.control is None:
                raise ValueError(f'Gate {gate.name} requires a control qubit')
                
        return v

class QuantumBenchmarkRequest(BaseModel):
    """
    Request model for quantum benchmarking API.
    Validates qubit count and benchmark options.
    """
    max_qubits: int = Field(
        150, 
        ge=2, 
        le=150, 
        description="Maximum number of qubits for the benchmark (2-150)",
        example=150
    )
    run_full_benchmark: bool = Field(
        False, 
        description="Whether to run the complete benchmark suite",
        example=True
    )
