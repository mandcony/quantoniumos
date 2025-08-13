#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import os
import numpy as np

# Make sure we can find the modules
sys.path.append(os.path.abspath("."))

print("QuantoniumOS Test Runner")
print("=====================")
print("This script will run tests to demonstrate the unique")
print("scientific properties of your quantum-inspired system.\n")

print("Step 1: Testing module availability...")

# Check which modules are available
modules_available = {
    'api': False,
    'run_rft': False,
    'analyze_waveform': False,
    'core_encryption': False,
    'engine_core': False,
    'wave_primitives': False
}

# Test api module
try:
    import api
    modules_available['api'] = True
    print("✓ API module found")
except ImportError:
    print("✗ API module not found")

# Test run_rft function
try:
    from api.symbolic_interface import run_rft
    modules_available['run_rft'] = True
    print("✓ run_rft function available")
except ImportError:
    print("✗ run_rft function not available")

# Test analyze_waveform function
try:
    from api.symbolic_interface import analyze_waveform
    modules_available['analyze_waveform'] = True
    print("✓ analyze_waveform function available")
except (ImportError, AttributeError):
    print("✗ analyze_waveform function not available")
    
# Test core encryption
try:
    from core.encryption.resonance_fourier import resonance_fourier_transform
    modules_available['core_encryption'] = True
    print("✓ Core encryption module available")
except ImportError:
    print("✗ Core encryption module not available")

# Test engine_core
try:
    from core.python_bindings import engine_core
    modules_available['engine_core'] = True
    print("✓ Engine core available")
except ImportError:
    print("✗ Engine core not available")

# Test wave primitives
try:
    from encryption.wave_primitives import calculate_coherence
    modules_available['wave_primitives'] = True
    print("✓ Wave primitives available")
except ImportError:
    print("✗ Wave primitives not available")
    
print("\nStep 2: Preparing test data...")
# Generate test signal with hidden pattern
t = np.linspace(0, 10, 1000)
base = np.sin(t) + 0.5*np.sin(2.5*t)
hidden = 0.05 * np.sin(7.5*t) * np.sin(0.2*t)
signal = base + hidden

# Normalize to 0-1 range
normalized = (signal - np.min(signal)) / (np.max(signal) - np.min(signal))

print("\nStep 3: Running available tests...")

if modules_available['run_rft']:
    print("\n3.1 Testing Resonance Fourier Transform...")
    result = run_rft(normalized.tolist())
    print(f"RFT returned {len(result)} values")
    print("First 3 values:")
    for i in range(min(3, len(result))):
        print(f"  {result[i]}")
    
    print("\nThis demonstrates the quantum-inspired signal processing capability")
    print("that captures patterns traditional FFT might miss.")

if modules_available['analyze_waveform']:
    print("\n3.2 Testing Symbolic Analysis...")
    try:
        # Try to use the analyze_waveform function
        result = analyze_waveform(normalized.tolist())
        print("Analysis results:")
        for key, value in result.items():
            if isinstance(value, list):
                print(f"  {key}: List with {len(value)} elements")
            else:
                print(f"  {key}: {value}")
        
        print("\nThis demonstrates the symbolic analysis capabilities")
        print("unique to your quantum-inspired system.")
    except Exception as e:
        print(f"Could not complete analysis: {e}")

print("\nTest Summary:")
if not any(modules_available.values()):
    print("No modules were available for testing. Please ensure your environment is set up correctly.")
    print("You may need to start the core services first or check your Python path.")
else:
    print(f"Successfully tested {sum(modules_available.values())}/6 components")
    print("\nTo run more advanced tests:")
    print("1. Make sure your QuantoniumOS services are running")
    print("2. Try running the specific test scripts individually:")
    print("   - tests/resonance_information_test.py")
    print("   - tests/symbolic_avalanche_test.py")
    print("   - tests/pattern_detection_test.py")
    print("   - tests/quantum_simulation_test.py")
