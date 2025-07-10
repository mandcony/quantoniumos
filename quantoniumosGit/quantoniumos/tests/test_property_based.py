"""
Property-based tests for core Quantonium OS algorithms using Hypothesis.
"""

import pytest
from hypothesis import given, strategies as st
from encryption.geometric_waveform_hash import geometric_waveform_hash
from encryption.resonance_fourier import resonance_fourier_transform

# Strategy for generating a list of floats between 0.0 and 1.0
waveform_strategy = st.lists(st.floats(min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False), min_size=4, max_size=128)

@given(waveform=waveform_strategy)
def test_rft_handles_various_waveforms(waveform):
    """
    Tests that the RFT function runs without errors for a wide range of valid inputs.
    """
    try:
        result = resonance_fourier_transform(waveform)
        # Result should be a dictionary with specific keys
        assert isinstance(result, dict)
        assert "frequencies" in result
        assert "amplitudes" in result
        assert "phases" in result
        assert "original_length" in result
        assert len(result["amplitudes"]) == (len(waveform) // 2) + 1
    except Exception as e:
        pytest.fail(f"resonance_fourier_transform failed with valid input: {e}")

# Strategy for generating valid inputs for the geometric hash
waveform_strategy_for_hash = st.lists(st.floats(min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False), min_size=1, max_size=1024)
nonce_strategy = st.binary(min_size=16, max_size=16)

@given(
    waveform=waveform_strategy_for_hash,
    nonce=nonce_strategy
)
def test_geometric_hash_handles_various_inputs(waveform, nonce):
    """
    Tests that the geometric_waveform_hash runs without errors for various inputs.
    """
    try:
        result = geometric_waveform_hash(waveform, nonce=nonce)
        # The result should be a base64-encoded string
        assert isinstance(result, str)
        # It should not be empty
        assert result
    except Exception as e:
        pytest.fail(f"geometric_waveform_hash failed with valid input: {e}")

@given(waveform=waveform_strategy_for_hash)
def test_geometric_hash_is_deterministic_with_same_inputs(waveform):
    """
    Tests that the hash function is deterministic: the same input always
    produces the same output.
    """
    nonce = b'a_fixed_nonce123' # Use a fixed nonce
    
    hash1 = geometric_waveform_hash(waveform, nonce=nonce)
    hash2 = geometric_waveform_hash(waveform, nonce=nonce)
    
    assert hash1 == hash2
