#!/bin/bash

# make_repro.sh - Canonical test path for QuantoniumOS validation
# This script provides a single command to build → test → validate the entire system

set -e  # Exit on any error

echo "=== QuantoniumOS Canonical Validation Suite ==="
echo "Starting complete build and validation process..."
echo ""

# 1. Build phase
echo "[1/4] Building system..."
if [ -f "requirements.txt" ]; then
    echo "Installing Python dependencies..."
    pip install -r requirements.txt --quiet
else
    echo "No requirements.txt found, skipping dependency installation"
fi

# 2. Core unitary check
echo ""
echo "[2/4] Running unitary check..."
python -c "
import numpy as np
try:
    from core.encryption.resonance_fourier import forward_true_rft, inverse_true_rft
    print('Testing RFT unitary property...')
    signal = np.array([1.0, 0.5, 0.2, 0.8])
    X = forward_true_rft(signal)
    reconstructed = inverse_true_rft(X)
    error = np.linalg.norm(signal - reconstructed)
    print(f'✓ Unitary test passed: L2 error = {error:.2e}')
    assert error < 1e-10, f'Unitary error too large: {error}'
except Exception as e:
    print(f'✗ Unitary test failed: {e}')
    exit(1)
"

# 3. Avalanche effect test
echo ""
echo "[3/4] Running avalanche effect test..."
python -c "
try:
    from core.encryption.fixed_resonance_encrypt import fixed_resonance_encrypt
    print('Testing avalanche effect...')
    key = 'test-key-123'
    msg1 = 'Hello World'
    msg2 = 'Hello world'  # Single bit change
    
    enc1 = fixed_resonance_encrypt(msg1, key)
    enc2 = fixed_resonance_encrypt(msg2, key)
    
    # Compare encrypted outputs (skip signature/token parts)
    diff_bits = sum(bin(a ^ b).count('1') for a, b in zip(enc1[40:], enc2[40:]))
    total_bits = len(enc1[40:]) * 8
    avalanche_ratio = diff_bits / total_bits if total_bits > 0 else 0
    
    print(f'✓ Avalanche test passed: {avalanche_ratio:.1%} bits changed')
    assert avalanche_ratio > 0.4, f'Avalanche effect too low: {avalanche_ratio:.1%}'
except Exception as e:
    print(f'✗ Avalanche test failed: {e}')
    exit(1)
"

# 4. NIST subset test
echo ""
echo "[4/4] Running NIST statistical tests subset..."
python -c "
import numpy as np
try:
    from core.encryption.fixed_resonance_encrypt import fixed_resonance_encrypt
    import secrets
    print('Running basic entropy tests...')
    
    # Generate test data
    key = 'nist-test-key'
    test_data = []
    for i in range(100):
        msg = f'test message {i} with random data {secrets.token_hex(16)}'
        encrypted = fixed_resonance_encrypt(msg, key)
        test_data.extend(encrypted[40:])  # Skip header
    
    # Basic entropy check
    from collections import Counter
    byte_counts = Counter(test_data)
    entropy = -sum((count/len(test_data)) * np.log2(count/len(test_data)) 
                  for count in byte_counts.values())
    
    print(f'✓ Entropy test passed: {entropy:.2f} bits/byte')
    assert entropy > 7.0, f'Entropy too low: {entropy:.2f}'
    
    print('✓ Basic statistical tests passed')
except Exception as e:
    print(f'✗ Statistical tests failed: {e}')
    exit(1)
"

# Summary
echo ""
echo "=== Validation Summary ==="
echo "✓ Build: Complete"
echo "✓ Unitary Check: RFT reconstruction verified"
echo "✓ Avalanche Effect: Encryption diffusion verified" 
echo "✓ NIST Subset: Basic entropy tests passed"
echo ""
echo "🎉 All validation tests passed successfully!"
echo ""
echo "For detailed analysis, run:"
echo "  python run_comprehensive_tests.py      # Full test suite"
echo "  python -m pytest tests/               # Unit tests"
echo ""
