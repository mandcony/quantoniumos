"""
Quantonium OS - API Data Models

Defines Pydantic-style request schemas for symbolic endpoints.
"""

from pydantic import BaseModel, Field
from typing import List

class EncryptRequest(BaseModel):
    plaintext: str = Field(..., example="hello world")
    key: str = Field(..., example="symbolic-key")

class RFTRequest(BaseModel):
    waveform: List[float] = Field(..., example=[0.1, 0.5, 0.9])

class EntropyRequest(BaseModel):
    amount: int = Field(32, ge=1, le=1024, example=64)

class DecryptRequest(BaseModel):
    ciphertext: str = Field(..., example="base64_encoded_ciphertext_string")
    key: str = Field(..., example="symbolic-key")

class ContainerUnlockRequest(BaseModel):
    waveform: List[float] = Field(..., example=[0.2, 0.7, 0.3])
    hash: str = Field(..., example="d6a88f4f...")
    key: str = Field(..., example="symbolic-key")

# Auto-unlock request model removed as requested - proper security model enforces both hash and key requirement
