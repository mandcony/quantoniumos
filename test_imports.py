#!/usr/bin/env python3
"""Test that all test file imports are working correctly."""

import sys

print("=" * 70)
print("Testing Import Routing")
print("=" * 70)

# Test 1: RFT Vertex Codec
print("\n[1] Testing rft_vertex_codec import...")
try:
    from algorithms.rft.compression import rft_vertex_codec as codec
    print("    ✓ Import successful")
    print(f"    Available functions: {[x for x in dir(codec) if not x.startswith('_')][:5]}")
except ImportError as e:
    print(f"    ✗ Import failed: {e}")

# Test 2: Pattern Editor (PyQt5)
print("\n[2] Testing pattern_editor import...")
try:
    from src.apps.quantsounddesign.pattern_editor import DrumSynthesizer, DrumType, HAS_PYQT5
    print(f"    ✓ Import successful")
    print(f"    PyQt5 available: {HAS_PYQT5}")
    print(f"    DrumSynthesizer: {DrumSynthesizer}")
except Exception as e:
    print(f"    ✗ Import failed: {e}")

# Test 3: Bell Violations Test
print("\n[3] Testing bell_violations imports...")
try:
    from tests.validation.test_bell_violations import EntangledVertexEngine, OpenQuantumSystem
    print("    ✓ Import successful (using stub classes)")
    print(f"    EntangledVertexEngine: {EntangledVertexEngine}")
except Exception as e:
    print(f"    ✗ Import failed: {e}")

# Test 4: Entanglement Protocols
print("\n[4] Testing entanglement_protocols import...")
try:
    from tests.proofs.test_entanglement_protocols import BellTestProtocol
    print("    ✓ Import successful")
    print(f"    BellTestProtocol: {BellTestProtocol}")
except Exception as e:
    print(f"    ✗ Import failed: {e}")

# Test 5: Hypothesis for property testing
print("\n[5] Testing hypothesis import...")
try:
    import hypothesis
    print(f"    ✓ Import successful")
    print(f"    Hypothesis version: {hypothesis.__version__}")
except ImportError:
    print(f"    ✗ Hypothesis not installed")

print("\n" + "=" * 70)
print("Import Testing Complete")
print("=" * 70)
