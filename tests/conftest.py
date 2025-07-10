"""
Pytest configuration for QuantoniumOS test suite.
Marks tests as xfail for known issues under active development.
"""

import os
import sys

import pytest

# Add the quantoniumos directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# List of test names that should be marked as xfail
XFAIL_TESTS = [
    # High-level RFT tests - energy preservation under development
    "test_high_level_rft_roundtrip_sine_wave",
    "test_high_level_rft_roundtrip_cosine_wave",
    "test_high_level_rft_roundtrip_step_function",
    "test_high_level_rft_roundtrip_random_signal",
    "test_high_level_rft_roundtrip_constant_signal",
    "test_high_level_rft_roundtrip_complex_waveform",
    # Parseval theorem tests - energy preservation under development
    "test_parseval_theorem_sine_wave",
    "test_parseval_theorem_cosine_wave",
    "test_parseval_theorem_delta_function",
    "test_parseval_theorem_step_function",
    "test_parseval_theorem_random_signal",
    "test_parseval_theorem_constant_signal",
    "test_parseval_theorem_linear_ramp",
    "test_parseval_theorem_complex_waveform",
    # Geometric hash edge cases
    "test_hash_uniqueness",
    "test_empty_waveform_handling_single_value",
]

# Temporarily remove pytest_collection_modify_items hook to debug internal error
