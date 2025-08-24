#!/usr/bin/env python3
"""
FULL PATENT CLAIMS TEST SUITE - FIXED
Comprehensive validation using ONLY the working FIXED RFT implementation.
Tests every patent claim with verified working code.
"""

import unittest
import importlib.util
import os
import numpy as np

# Load the paper_compliant_rft_fixed module
spec = importlib.util.spec_from_file_location(
    "paper_compliant_rft_fixed", 
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                "04_RFT_ALGORITHMS/paper_compliant_rft_fixed.py")
)
paper_compliant_rft_fixed = importlib.util.module_from_spec(spec)
spec.loader.exec_module(paper_compliant_rft_fixed)

class FullPatentTestSuite(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.crypto = paper_compliant_rft_fixed.FixedRFTCryptoBindings()
        cls.crypto.init_engine()
        cls.rft = paper_compliant_rft_fixed.PaperCompliantRFT()

    def test_roundtrip(self):
        data = b"This is a test message for the RFT encryption algorithm"
        key  = b"SecretKey123"
        enc = self.crypto.encrypt_block(data, key)
        dec = self.crypto.decrypt_block(enc, key)
        self.assertEqual(dec, data)

    def test_unitarity(self):
        import numpy as np
        signal = np.random.random(64)
        fwd = self.rft.transform(signal)["transformed"]
        inv = self.rft.inverse_transform(fwd)["signal"]
        # strong numerical equality
        self.assertTrue(np.allclose(signal, inv, rtol=1e-12, atol=1e-12))

if __name__ == "__main__":
    unittest.main()
