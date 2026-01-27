# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Luis M. Minier / quantoniumos
import secrets
import pytest
from algorithms.rft.crypto.enhanced_cipher import EnhancedRFTCryptoV2
from hypothesis import given, settings, strategies as st

@given(st.binary(min_size=16, max_size=1024))
@settings(deadline=None)  # Disable deadline for slow crypto operations
def test_encryption_roundtrip_property(data):
    key = secrets.token_bytes(32)
    cipher = EnhancedRFTCryptoV2(key)
    ct = cipher.encrypt_aead(data, b"prop")
    pt = cipher.decrypt_aead(ct, b"prop")
    assert pt == data
