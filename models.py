from typing import Dict, List, Optional, Any, Union
from pydantic import BaseModel, Field

# Request and Response Models for /encrypt endpoint
class EncryptRequest(BaseModel):
    plaintext: str = Field(..., description="Text to be encrypted")
    key: str = Field(..., description="Encryption key")

class EncryptResponse(BaseModel):
    ciphertext: str = Field(..., description="Encrypted text")
    status: str = Field("success", description="Response status")
    timestamp: Optional[int] = None
    signature: Optional[str] = None

# Request and Response Models for /simulate/rft endpoint
class RFTRequest(BaseModel):
    waveform: List[float] = Field(..., description="Input waveform for Resonance Fourier Transform")

class RFTResponse(BaseModel):
    frequency_data: Dict[str, float] = Field(..., description="Frequency domain data from RFT")
    status: str = Field("success", description="Response status")
    timestamp: Optional[int] = None
    signature: Optional[str] = None

# Request and Response Models for /entropy/sample endpoint
class EntropySampleRequest(BaseModel):
    amount: int = Field(32, description="Amount of entropy bytes to generate", ge=1, le=1024)

class EntropySampleResponse(BaseModel):
    entropy: str = Field(..., description="Base64 encoded entropy data")
    status: str = Field("success", description="Response status")
    timestamp: Optional[int] = None
    signature: Optional[str] = None

# Request and Response Models for /container/unlock endpoint
class ContainerUnlockRequest(BaseModel):
    waveform: List[float] = Field(..., description="Symbolic waveform for container unlock")
    hash: str = Field(..., description="Hash value for verification")

class ContainerUnlockResponse(BaseModel):
    unlocked: bool = Field(..., description="Whether the container was successfully unlocked")
    status: str = Field("success", description="Response status")
    timestamp: Optional[int] = None
    signature: Optional[str] = None
