"""
Property-based tests for core Quantonium OS algorithms using Hypothesis.
"""

import pytest
from hypothesis import given, strategies as st
from encryption.geometric_waveform_hash import geometric_waveform_hash
from encryption.resonance_fourier import resonance_fourier_transform

# Strategy for generating a list of floats between 0.0 and 1.0
waveform_strategy = st.lists(st.floats(min_value=0.0, max_value=1.0), min_size=4, max_size=128)

@given(waveform=waveform_strategy)
def test_rft_handles_various_waveforms(waveform):
    """
    Tests that the RFT function runs without errors for a wide range of valid inputs.
    """
    try:
        result = resonance_fourier_transform(waveform)
        # Result should be a list of floats
        assert isinstance(result, list)
        assert all(isinstance(x, float) for x in result)
        # RFT should produce output of the same length as the input
        assert len(result) == len(waveform)
    except Exception as e:
        pytest.fail(f"resonance_fourier_transform failed with valid input: {e}")

# Strategy for generating valid plaintext and keys for the geometric hash
plaintext_strategy = st.text(min_size=1, max_size=1024)
key_strategy = st.text(min_size=4, max_size=64)
nonce_strategy = st.binary(min_size=16, max_size=16)

@given(
    plaintext=plaintext_strategy,
    key=key_strategy,
    nonce=nonce_strategy
)
def test_geometric_hash_handles_various_inputs(plaintext, key, nonce):
    """
    Tests that the geometric_waveform_hash runs without errors for various inputs.
    """
    try:
        result = geometric_waveform_hash(plaintext, key, nonce)
        # The result should be a base64-encoded string
        assert isinstance(result, str)
        # It should not be empty
        assert result
    except Exception as e:
        pytest.fail(f"geometric_waveform_hash failed with valid input: {e}")

@given(data=st.binary())
def test_geometric_hash_is_deterministic_with_same_inputs(data):
    """
    Tests that the hash function is deterministic: the same input always
    produces the same output.
    """
    plaintext = data.decode('utf-8', errors='ignore')
    if not plaintext: # Skip empty plaintext generated from invalid bytes
        return

    key = "deterministic-key"
    nonce = b'a_fixed_nonce123'
    
    hash1 = geometric_waveform_hash(plaintext, key, nonce)
    hash2 = geometric_waveform_hash(plaintext, key, nonce)
    
    assert hash1 == hash2
