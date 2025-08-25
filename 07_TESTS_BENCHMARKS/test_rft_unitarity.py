#!/usr/bin/env python3
"""
Unitarity test for paper_compliant_rft_fixed.py
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), '04_RFT_ALGORITHMS'))
import paper_compliant_rft_fixed as paper_compliant_rft_fixedimport numpy as np

def test_rft_unitarity():
    """Test the unitarity property of the RFT transform"""
    print('Testing RFT unitarity with different sizes...')
    print('=' * 70)
    print('{0:>6} | {1:>10} | {2:>10} | {3:>12} | {4:>8}'.format(
        'Size', 'Max Error', 'Mean Error', 'Power Ratio', 'Unitary'))
    print('-' * 70)

    for size in [16, 32, 64]:
        rft = paper_compliant_rft_fixed.PaperCompliantRFT(size=size)

        # Generate random signal
        signal = np.random.random(size)

        # Apply transform
        result = rft.transform(signal)

        # Apply inverse transform
        inverse = rft.inverse_transform(result['transformed'])

        # Check if power is preserved (Parseval's theorem)
        input_power = np.sum(np.abs(signal)**2)
        output_power = np.sum(np.abs(result['transformed'])**2)
        power_ratio = output_power / input_power if input_power > 0 else 0

        # Check roundtrip error
        max_error = np.max(np.abs(signal - inverse['signal']))
        mean_error = np.mean(np.abs(signal - inverse['signal']))

        # Check if the transform is unitary
        # (Input and output should have same energy)
        is_unitary = np.isclose(input_power, output_power, rtol=1e-5)

        print('{0:>6} | {1:>10.4e} | {2:>10.4e} | {3:>12.6f} | {4:>8}'.format(
            size, max_error, mean_error, power_ratio, str(is_unitary)))

    print('=' * 70)

if __name__ == "__main__":
    print("Running RFT Unitarity Test\n")
    test_rft_unitarity()
    print("\nTest completed.")
