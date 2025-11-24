"""Large-scale tests for EnhancedRFTCryptoV2 encryption routines.

Notes:
- The megabyte-payload AEAD test is marked as `slow` and the payload size can be
    overridden for local quick runs using the environment variable
    `RFT_CRYPTO_TEST_PAYLOAD` (bytes). Example: set to 131072 for 128KB.
"""
from __future__ import annotations

import os
import secrets

import numpy as np

from algorithms.crypto.rft.enhanced_cipher import EnhancedRFTCryptoV2
import pytest


def test_aead_roundtrip_fast_payload_64kb() -> None:
    """Quick sanity AEAD roundtrip with ~64KB to keep local runs fast."""
    key = secrets.token_bytes(32)
    cipher = EnhancedRFTCryptoV2(key)
    payload = os.urandom(64 * 1024)
    aad = b"fast-payload-meta"

    encrypted = cipher.encrypt_aead(payload, aad)
    decrypted = cipher.decrypt_aead(encrypted, aad)

    assert decrypted == payload
    assert len(encrypted) > len(payload)


@pytest.mark.slow
def test_aead_roundtrip_megabyte_payload() -> None:
    key = secrets.token_bytes(32)
    cipher = EnhancedRFTCryptoV2(key)
    # Allow overriding payload size for quicker local runs
    payload_size = int(os.environ.get("RFT_CRYPTO_TEST_PAYLOAD", "1048576"))
    payload = os.urandom(payload_size)  # default 1 MB
    aad = b"megabyte-payload-meta"

    encrypted = cipher.encrypt_aead(payload, aad)
    decrypted = cipher.decrypt_aead(encrypted, aad)

    assert decrypted == payload
    assert len(encrypted) > len(payload)


def test_aead_multiple_blocks_no_reuse() -> None:
    key = secrets.token_bytes(32)
    cipher = EnhancedRFTCryptoV2(key)
    payload = os.urandom(512_000)

    ct1 = cipher.encrypt_aead(payload)
    ct2 = cipher.encrypt_aead(payload)

    assert ct1 != ct2  # random salt must differentiate outputs


def test_aead_bit_error_diffusion() -> None:
    key = secrets.token_bytes(32)
    cipher = EnhancedRFTCryptoV2(key)
    payload = os.urandom(131_072)

    ct = cipher.encrypt_aead(payload)
    tampered = bytearray(ct)
    tampered[len(tampered) // 2] ^= 0x01

    try:
        cipher.decrypt_aead(bytes(tampered))
    except ValueError:
        return
    raise AssertionError("Authentication should fail when ciphertext is modified")
