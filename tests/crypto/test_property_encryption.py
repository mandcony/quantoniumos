import secrets
import pytest
from algorithms.rft.core.enhanced_rft_crypto_v2 import EnhancedRFTCryptoV2
from hypothesis import given, strategies as st

@given(st.binary(min_size=16, max_size=1024))
def test_encryption_roundtrip_property(data):
    key = secrets.token_bytes(32)
    cipher = EnhancedRFTCryptoV2(key)
    ct = cipher.encrypt_aead(data, b"prop")
    pt = cipher.decrypt_aead(ct, b"prop")
    assert pt == data
