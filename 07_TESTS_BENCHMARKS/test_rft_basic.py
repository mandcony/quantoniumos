#!/usr/bin/env python3
"""
Basic test for paper_compliant_rft_fixed.py
"""

import sys
import os
from importlib import import_module
import numpy as np

# Add the root directory to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the module
paper_compliant_rft_fixed = import_module("04_RFT_ALGORITHMS.paper_compliant_rft_fixed")

def test_simple_encryption():
    """Test basic encryption and decryption roundtrip"""
    print('\nTesting encryption/decryption roundtrip...')
    crypto = paper_compliant_rft_fixed.PaperCompliantRFT()
    crypto.init_engine()
    
    test_data = b'This is a test message for the RFT encryption algorithm'
    key = b'SecretKey123'

    encrypted = crypto.encrypt_block(test_data, key)
    decrypted = crypto.decrypt_block(encrypted, key)

    print(f'Original:  {test_data}')
    print(f'Decrypted: {decrypted}')
    print(f'Roundtrip Successful: {test_data == decrypted}')

if __name__ == "__main__":
    print("Running Basic RFT Test\n")
    test_simple_encryption()
    print("\nTest completed.")
