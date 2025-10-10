"""Large-scale tests for EnhancedRFTCryptoV2 encryption routines."""
from __future__ import annotations

import os
import secrets

import numpy as np

from core.enhanced_rft_crypto_v2 import EnhancedRFTCryptoV2


def test_aead_roundtrip_megabyte_payload() -> None:
    key = secrets.token_bytes(32)
    cipher = EnhancedRFTCryptoV2(key)
    payload = os.urandom(1_048_576)  # 1 MB random input
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
