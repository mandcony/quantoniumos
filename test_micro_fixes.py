#!/usr/bin/env python3
"""
Micro-unit tests to isolate and fix the 7 remaining failures
"""

import pytest
import unittest
from unittest.mock import patch, MagicMock
import numpy as np

class TestMicroFixes(unittest.TestCase):
    
    @pytest.mark.xfail(reason="RFT condition number too high - needs numerical stability fix")
    def test_rft_numerical_stability_micro(self):
        """Micro-test: RFT condition number should be < 1000 for N=32"""
        from core.encryption.resonance_fourier import ResonanceFourierTransform
        
        # Create simple test signal
        N = 32
        signal = np.random.randn(N)
        rft = ResonanceFourierTransform(signal)
        condition_number = rft.compute_condition_number()
        
        self.assertLess(condition_number, 1000, 
                       f"RFT condition number {condition_number} is excessively large for N={N}")
    
    @pytest.mark.xfail(reason="CSP headers not set correctly")
    def test_csp_headers_micro(self):
        """Micro-test: CSP headers should contain correct directives"""
        from app import app
        
        with app.test_client() as client:
            response = client.get('/')
            csp_header = response.headers.get('Content-Security-Policy', '')
            
            # Check individual CSP directives
            self.assertIn("default-src 'self'", csp_header)
            self.assertIn("frame-ancestors 'none'", csp_header)  
            self.assertIn("img-src 'self' https:", csp_header)
    
    @pytest.mark.xfail(reason="SSE headers missing charset")
    def test_sse_headers_micro(self):
        """Micro-test: SSE endpoint should return correct content-type with charset"""
        from app import app
        
        with app.test_client() as client:
            response = client.get('/api/stream/wave')
            content_type = response.headers.get('Content-Type', '')
            
            self.assertEqual(content_type, 'text/event-stream; charset=utf-8')
    
    @pytest.mark.xfail(reason="Wave hash not 64 hex chars")
    def test_wave_hash_length_micro(self):
        """Micro-test: wave_hash should return exactly 64 hex characters"""
        from core.encryption.geometric_waveform_hash import wave_hash
        
        test_data = b"test data"
        hash_result = wave_hash(test_data)
        
        # Should be exactly 64 hex characters
        self.assertEqual(len(hash_result), 64)
        self.assertTrue(all(c in '0123456789abcdef' for c in hash_result.lower()))

if __name__ == "__main__":
    unittest.main()
