"""
Quantonium OS - API Data Models

Defines Pydantic-style request schemas for symbolic endpoints and quantum API.
Includes comprehensive validation for all API inputs.
"""

import base64
import re
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field, field_validator

# Maximum length constraints for security
MAX_KEY_LENGTH = 256
MAX_PLAINTEXT_LENGTH = 8192  # 8KB
MAX_CIPHERTEXT_LENGTH = 1024
MAX_WAVEFORM_LENGTH = 64
MIN_WAVEFORM_LENGTH = 3

# Pattern validation for specific fields
HASH_PATTERN = r"^[A-Za-z0-9+/=]+$"  # Base64 pattern for container hashes


class EncryptRequest(BaseModel):
    """Request model for encrypting data using resonance techniques."""

    plaintext: str = Field(..., example="hello world", description="Text to encrypt")
    key: str = Field(..., example="symbolic-key", description="Encryption key")

    @field_validator("plaintext")
    @classmethod
    def validate_plaintext(cls, v):
        if not v.strip():
            raise ValueError("Plaintext cannot be empty or whitespace")
        if len(v) > MAX_PLAINTEXT_LENGTH:
            raise ValueError(
                f"Plaintext too long (max {MAX_PLAINTEXT_LENGTH} characters)"
            )
        return v

    @field_validator("key")
    @classmethod
    def validate_key(cls, v):
        if not v.strip():
            raise ValueError("Key cannot be empty or whitespace")
        if len(v) < 4:
            raise ValueError("Key must be at least 4 characters long")
        if len(v) > MAX_KEY_LENGTH:
            raise ValueError(f"Key too long (max {MAX_KEY_LENGTH} characters)")
        return v


class RFTRequest(BaseModel):
    """Request model for Resonance Fourier Transform operations."""

    waveform: List[float] = Field(
        ...,
        example=[0.1, 0.5, 0.9],
        description="Waveform data as a list of floating point values",
    )

    @field_validator("waveform")
    @classmethod
    def validate_waveform(cls, v):
        # Check length
        if len(v) < MIN_WAVEFORM_LENGTH:
            raise ValueError(
                f"Waveform must have at least {MIN_WAVEFORM_LENGTH} values"
            )
        if len(v) > MAX_WAVEFORM_LENGTH:
            raise ValueError(f"Waveform too long (max {MAX_WAVEFORM_LENGTH} values)")

        # Ensure all values are in range [0.0, 1.0]
        for val in v:
            if val < 0.0 or val > 1.0:
                raise ValueError(
                    f"Waveform values must be between 0.0 and 1.0, got {val}"
                )
        return v


class EntropyRequest(BaseModel):
    """Request model for generating quantum-inspired entropy."""

    amount: int = Field(
        32,
        ge=1,
        le=1024,
        example=64,
        description="Amount of entropy to generate in bytes",
    )


class DecryptRequest(BaseModel):
    """Request model for decrypting data using resonance techniques."""

    ciphertext: str = Field(
        ...,
        example="Pm0C/vQlcaH4OL71EYN5zlIcWyltZNnjlD6XQqZwTLvLSkwfu2xvUA==",
        description="Ciphertext to decrypt (base64 encoded)",
    )
    key: str = Field(..., example="symbolic-key", description="Decryption key")

    @field_validator("ciphertext")
    @classmethod
    def validate_ciphertext(cls, v):
        if not v:
            raise ValueError("Ciphertext cannot be empty")
        if len(v) > MAX_CIPHERTEXT_LENGTH:
            raise ValueError(
                f"Ciphertext too long (max {MAX_CIPHERTEXT_LENGTH} characters)"
            )
        if not re.match(HASH_PATTERN, v):
            raise ValueError("Ciphertext contains invalid characters")

        # Validate that it's properly base64 encoded
        try:
            # Just checking format, not actually using the decoded value
            base64.b64decode(v)
        except Exception:
            raise ValueError("Ciphertext must be valid base64 encoded data")
        return v

    @field_validator("key")
    @classmethod
    def validate_key(cls, v):
        if not v.strip():
            raise ValueError("Key cannot be empty or whitespace")
        if len(v) < 4:
            raise ValueError("Key must be at least 4 characters long")
        if len(v) > MAX_KEY_LENGTH:
            raise ValueError(f"Key too long (max {MAX_KEY_LENGTH} characters)")
        return v


class ContainerUnlockRequest(BaseModel):
    """Request model for unlocking symbolic containers."""

    waveform: List[float] = Field(
        ...,
        example=[0.2, 0.7, 0.3],
        description="Waveform data as a list of floating point values",
    )
    hash: str = Field(
        ...,
        example="Pm0C/vQlcaH4OL71EYN5zlIcWyltZNnjlD6XQqZwTLvLSkwfu2xvUA==",
        description="Container hash identifier",
    )
    key: str = Field(..., example="symbolic-key", description="Decryption key")

    @field_validator("waveform")
    @classmethod
    def validate_waveform(cls, v):
        # Check length
        if len(v) < MIN_WAVEFORM_LENGTH:
            raise ValueError(
                f"Waveform must have at least {MIN_WAVEFORM_LENGTH} values"
            )
        if len(v) > MAX_WAVEFORM_LENGTH:
            raise ValueError(f"Waveform too long (max {MAX_WAVEFORM_LENGTH} values)")

        # Ensure all values are in range [0.0, 1.0]
        for val in v:
            if val < 0.0 or val > 1.0:
                raise ValueError(
                    f"Waveform values must be between 0.0 and 1.0, got {val}"
                )
        return v

    @field_validator("hash")
    @classmethod
    def validate_hash(cls, v):
        if not v:
            raise ValueError("Hash cannot be empty")
        if len(v) > MAX_CIPHERTEXT_LENGTH:
            raise ValueError(f"Hash too long (max {MAX_CIPHERTEXT_LENGTH} characters)")
        if not re.match(HASH_PATTERN, v):
            raise ValueError("Hash contains invalid characters")
        return v

    @field_validator("key")
    @classmethod
    def validate_key(cls, v):
        if not v.strip():
            raise ValueError("Key cannot be empty or whitespace")
        if len(v) < 4:
            raise ValueError("Key must be at least 4 characters long")
        if len(v) > MAX_KEY_LENGTH:
            raise ValueError(f"Key too long (max {MAX_KEY_LENGTH} characters)")
        return v


class SignRequest(BaseModel):
    """Request model for signing data using wave HMAC."""

    message: str = Field(..., example="message to sign", description="Message to sign")
    key: str = Field(..., example="symbolic-key", description="Signing key")
    use_phase_info: bool = Field(
        True,
        example=True,
        description="Whether to include phase information in signature",
    )

    @field_validator("message")
    @classmethod
    def validate_message(cls, v):
        if not v.strip():
            raise ValueError("Message cannot be empty or whitespace")
        if len(v) > MAX_PLAINTEXT_LENGTH:
            raise ValueError(
                f"Message too long (max {MAX_PLAINTEXT_LENGTH} characters)"
            )
        return v

    @field_validator("key")
    @classmethod
    def validate_key(cls, v):
        if not v.strip():
            raise ValueError("Key cannot be empty or whitespace")
        if len(v) < 4:
            raise ValueError("Key must be at least 4 characters long")
        if len(v) > MAX_KEY_LENGTH:
            raise ValueError(f"Key too long (max {MAX_KEY_LENGTH} characters)")
        return v


class VerifyRequest(BaseModel):
    """Request model for verifying signatures using wave HMAC."""

    message: str = Field(
        ..., example="message to verify", description="Original message"
    )
    signature: str = Field(
        ...,
        example="Pm0C/vQlcaH4OL71EYN5zlIcWyltZNnjlD6XQqZwTLvLSkwfu2xvUA==",
        description="Signature to verify (base64 encoded)",
    )
    key: str = Field(..., example="symbolic-key", description="Verification key")
    use_phase_info: bool = Field(
        True,
        example=True,
        description="Whether phase information was used in signature",
    )

    @field_validator("message")
    @classmethod
    def validate_message(cls, v):
        if not v.strip():
            raise ValueError("Message cannot be empty or whitespace")
        if len(v) > MAX_PLAINTEXT_LENGTH:
            raise ValueError(
                f"Message too long (max {MAX_PLAINTEXT_LENGTH} characters)"
            )
        return v

    @field_validator("signature")
    @classmethod
    def validate_signature(cls, v):
        if not v:
            raise ValueError("Signature cannot be empty")
        if len(v) > MAX_CIPHERTEXT_LENGTH:
            raise ValueError(
                f"Signature too long (max {MAX_CIPHERTEXT_LENGTH} characters)"
            )
        if not re.match(HASH_PATTERN, v):
            raise ValueError("Signature contains invalid characters")
        return v

    @field_validator("key")
    @classmethod
    def validate_key(cls, v):
        if not v.strip():
            raise ValueError("Key cannot be empty or whitespace")
        if len(v) < 4:
            raise ValueError("Key must be at least 4 characters long")
        if len(v) > MAX_KEY_LENGTH:
            raise ValueError(f"Key too long (max {MAX_KEY_LENGTH} characters)")
        return v


class IRFTRequest(BaseModel):
    """Request model for Inverse Resonance Fourier Transform operations."""

    frequency_data: Dict[str, Any] = Field(
        ...,
        example={
            "frequencies": [0.1, 0.3, 0.5],
            "amplitudes": [0.7, 0.3, 0.1],
            "phases": [0.0, 0.25, 0.5],
        },
        description="Frequency domain data with frequencies, amplitudes, and phases",
    )

    @field_validator("frequency_data")
    @classmethod
    def validate_frequency_data(cls, v):
        # Check required keys
        required_keys = ["frequencies", "amplitudes", "phases"]
        for key in required_keys:
            if key not in v:
                raise ValueError(f"Missing required key '{key}' in frequency_data")

        # Check that all arrays have the same length
        lengths = [len(v[key]) for key in required_keys]
        if not all(length == lengths[0] for length in lengths):
            raise ValueError("All frequency data arrays must have the same length")

        # Check values are in proper range
        for amp in v["amplitudes"]:
            if not 0.0 <= amp <= 1.0:
                raise ValueError(
                    f"Amplitude values must be between 0.0 and 1.0, got {amp}"
                )

        for phase in v["phases"]:
            if not 0.0 <= phase <= 1.0:
                raise ValueError(
                    f"Phase values must be between 0.0 and 1.0, got {phase}"
                )

        return v


# Structured quantum circuit validation
class Gate(BaseModel):
    """Model for a quantum gate definition."""

    name: str = Field(..., example="h")
    target: int = Field(..., example=0, ge=0)
    control: Optional[int] = Field(None, example=1, ge=0)


class QuantumCircuit(BaseModel):
    """Model for a complete quantum circuit definition."""

    gates: List[Gate] = Field(
        ...,
        example=[
            {"name": "h", "target": 0},
            {"name": "cnot", "control": 0, "target": 1},
        ],
    )


class QuantumCircuitRequest(BaseModel):
    """Request model for quantum circuit processing API."""

    circuit: Dict[str, Any] = Field(
        ...,
        example={
            "gates": [
                {"name": "h", "target": 0},
                {"name": "cnot", "control": 0, "target": 1},
            ]
        },
        description="Quantum circuit definition",
    )
    qubit_count: int = Field(
        3,
        ge=1,
        le=150,
        example=3,
        description="Number of qubits to use in the circuit (1-150)",
    )

    @field_validator("circuit")
    @classmethod
    def validate_circuit(cls, v, values):
        # Check that circuit has gates
        if "gates" not in v:
            raise ValueError("Circuit must contain 'gates' key")

        gates = v["gates"]
        if not isinstance(gates, list):
            raise ValueError("Gates must be a list")

        # Validate each gate
        for i, gate in enumerate(gates):
            if not isinstance(gate, dict):
                raise ValueError(f"Gate {i} must be a dictionary")

            # Required fields
            if "name" not in gate:
                raise ValueError(f"Gate {i} missing required field 'name'")
            if "target" not in gate:
                raise ValueError(f"Gate {i} missing required field 'target'")

            # Type checking
            if not isinstance(gate["name"], str):
                raise ValueError(f"Gate {i} 'name' must be a string")
            if not isinstance(gate["target"], int):
                raise ValueError(f"Gate {i} 'target' must be an integer")

            # Range checking if qubit_count is provided
            if "qubit_count" in values:
                qubit_count = values["qubit_count"]
                if gate["target"] >= qubit_count:
                    raise ValueError(
                        f"Gate {i} target {gate['target']} is out of range for qubit_count {qubit_count}"
                    )

                # Check control if present
                if "control" in gate:
                    if not isinstance(gate["control"], int):
                        raise ValueError(f"Gate {i} 'control' must be an integer")
                    if gate["control"] >= qubit_count:
                        raise ValueError(
                            f"Gate {i} control {gate['control']} is out of range for qubit_count {qubit_count}"
                        )

        return v


class QuantumBenchmarkRequest(BaseModel):
    """Request model for quantum benchmarking API."""

    max_qubits: int = Field(
        150,
        ge=1,
        le=150,
        example=150,
        description="Maximum number of qubits to benchmark (1-150)",
    )
    run_full_benchmark: bool = Field(
        False, example=True, description="Whether to run the full benchmark suite"
    )
