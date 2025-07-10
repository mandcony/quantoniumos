"""
QuantoniumOS - API Routes

This module defines FastAPI routes for the QuantoniumOS patent validation system.
All proprietary algorithm calls are properly encapsulated and never exposed.
"""

import hashlib
import logging
import os
import time
from datetime import datetime
from typing import Any, Dict, Optional

from fastapi import (APIRouter, Depends, File, Form, HTTPException, Request,
                     UploadFile)
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field, field_validator

from api.resonance_metrics import compute_avalanche, compute_rft
# Import internal modules
from api.symbolic_interface import get_symbolic_engine
from encryption.entropy_qrng import generate_entropy
from encryption.resonance_encrypt import encrypt_symbolic
from orchestration.resonance_manager import run_benchmark
from orchestration.symbolic_container import open_vault, validate_container

# Setup logging
log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_file = f"{log_dir}/session_{timestamp}.log"

logging.basicConfig(
    filename=log_file,
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("quantonium_api")

# Create API router
router = APIRouter()


# Pydantic models for request validation
class EncryptRequest(BaseModel):
    plaintext: str = Field(
        ..., min_length=32, max_length=32, description="128-bit hex plaintext"
    )
    key: str = Field(..., min_length=32, max_length=32, description="128-bit hex key")

    @field_validator("plaintext", "key")
    @classmethod
    def validate_hex(cls, v: str) -> str:
        # Ensure input is valid hexadecimal
        try:
            bytes.fromhex(v)
        except ValueError:
            raise ValueError("must be a valid 32-character hex string")
        return v


class AnalyzeRequest(BaseModel):
    payload: str = Field(..., description="Data to analyze with RFT")


# Security middleware to compute and add SHA-256 hash to responses
def add_signature(data: Dict[str, Any]) -> Dict[str, Any]:
    """Compute SHA-256 of payload and add as signature"""
    # Create a deterministic representation of the data
    serialized = str(sorted(data.items()))
    # Compute hash
    signature = hashlib.sha256(serialized.encode()).hexdigest()
    # Add signature to response
    data["signature"] = signature
    return data


def log_request(endpoint: str, request_data: Any = None, response_data: Any = None):
    """Log request and response details"""
    entry = {
        "timestamp": datetime.now().isoformat(),
        "endpoint": endpoint,
    }

    if request_data:
        if isinstance(request_data, dict) and "key" in request_data:
            # Don't log the actual key
            sanitized = request_data.copy()
            sanitized["key"] = "[REDACTED]"
            entry["request"] = sanitized
        else:
            entry["request"] = request_data

    if response_data:
        if isinstance(response_data, dict) and "signature" in response_data:
            entry["signature"] = response_data["signature"]

    logger.info(f"API Request: {entry}")


# Routes
@router.post("/encrypt")
async def encrypt(req: EncryptRequest):
    """
    Encrypt plaintext using resonance encryption

    Returns encrypted result with metrics:
    - hr: Harmonic Resonance
    - wc: Waveform Coherence
    - sa: Symbolic Alignment Vector
    - entropy: Entropy measurement
    - signature: SHA-256 integrity signature
    """
    log_request("/encrypt", req.dict())

    # Perform encryption with all metrics
    result = encrypt_symbolic(req.plaintext, req.key)

    # Add signature for integrity verification
    signed_result = add_signature(result)

    log_request("/encrypt", None, signed_result)
    return JSONResponse(content=signed_result)


@router.post("/benchmark")
async def benchmark():
    """
    Run the 64-test perturbation suite

    Streams CSV results of the benchmark tests showing:
    - test_id
    - bit_pos
    - perturb_type
    - hr (Harmonic Resonance)
    - wc (Waveform Coherence)
    - sa (Symbolic Alignment)
    - entropy
    - signature
    """
    log_request("/benchmark")

    # Create async generator for streaming response
    async def generate_csv():
        # CSV header
        yield "test_id,bit_pos,perturb_type,hr,wc,sa,entropy,signature\n"

        # Generate all benchmark rows
        for row in run_benchmark():
            # Convert benchmark row to CSV line
            line = ",".join(str(value) for value in row.values())
            yield f"{line}\n"

    # Set the latest benchmark to be accessible via /log/latest.csv
    log_request("/benchmark", None, {"status": "completed"})

    return StreamingResponse(
        generate_csv(),
        media_type="text/csv",
        headers={
            "Content-Disposition": f"attachment; filename=benchmark_{timestamp}.csv"
        },
    )


@router.post("/entropy")
async def entropy():
    """
    Generate a new quantum-inspired random entropy sample

    Returns:
    - entropy: Float entropy value
    - signature: SHA-256 integrity signature
    """
    log_request("/entropy")

    # Generate new entropy sample
    result = generate_entropy()

    # Add signature
    signed_result = add_signature(result)

    log_request("/entropy", None, signed_result)
    return JSONResponse(content=signed_result)


@router.post("/vault/open")
async def vault_open(
    file: UploadFile = File(...),
    wave_hash: str = Form(..., min_length=64, max_length=64),
):
    """
    Open a symbolic vault container using amplitude-phase waveform hash

    Returns:
    - status: "unlocked" or "denied"
    - wc: Waveform Coherence between provided hash and container
    - signature: SHA-256 integrity signature
    """
    log_request("/vault/open", {"filename": file.filename, "wave_hash": wave_hash})

    # Read uploaded file
    content = await file.read()

    # Try to open the vault with the provided wave hash
    result = open_vault(content, wave_hash)

    # Add signature
    signed_result = add_signature(result)

    log_request("/vault/open", None, signed_result)
    return JSONResponse(content=signed_result)


@router.post("/analyze")
async def analyze(req: AnalyzeRequest):
    """
    Perform full RFT + SA vector analysis on the provided payload

    Returns full RFT + SA vector dump as JSON
    """
    log_request("/analyze", {"payload_length": len(req.payload)})

    # Compute RFT analysis
    result = compute_rft(req.payload)

    # Add signature
    signed_result = add_signature(result)

    log_request("/analyze", None, signed_result)
    return JSONResponse(content=signed_result)


@router.get("/log/latest.csv")
async def latest_log():
    """Serve the latest benchmark CSV file"""
    log_request("/log/latest.csv")

    # Find the most recent log file
    log_files = sorted([f for f in os.listdir(log_dir) if f.endswith(".csv")])

    if not log_files:
        raise HTTPException(status_code=404, detail="No benchmark logs available")

    latest_log = os.path.join(log_dir, log_files[-1])

    def iterfile():
        with open(latest_log, "rb") as f:
            yield from f

    return StreamingResponse(
        iterfile(),
        media_type="text/csv",
        headers={"Content-Disposition": f"attachment; filename={log_files[-1]}"},
    )


@router.get("/entropy/stream")
async def entropy_stream():
    """Return last-N entropy samples for dashboard tiny-chart."""
    from pathlib import Path, json, time

    log_dir = Path("logs")
    latest = max(log_dir.glob("session_*.log"), key=lambda p: p.stat().st_mtime)
    ent = []
    with latest.open() as f:
        for line in f:
            if '"entropy"' in line:
                ent.append(json.loads(line)["entropy"])
    return {"series": ent[-50:], "t": time.time()}
