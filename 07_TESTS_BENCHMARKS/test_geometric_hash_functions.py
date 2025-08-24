import os
import sys
import unittest

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.dirname(os.path.dirname(__file__))))

from core.encryption.geometric_waveform_hash import generate_waveform_hash


class TestGeometricWaveformHash(unittest.TestCase):
    """Test that the geometric waveform hash module exports the expected functions."""

    def test_generate_waveform_hash(self):
        """Test that generate_waveform_hash exists and returns expected format."""
        # Create parameters
        A = 0.5
        phi = 0.25

        # Generate hash
        hash_result = generate_waveform_hash([A * phi for i in range(8)])

        # Should be a string
        self.assertIsInstance(hash_result, str)

        # Should not be empty
        self.assertTrue(len(hash_result) > 0)


if __name__ == "__main__":
    unittest.main()
