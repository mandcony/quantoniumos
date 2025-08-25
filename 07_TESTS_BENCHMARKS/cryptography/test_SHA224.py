# -*- coding: utf-8 -*-
#
# QuantoniumOS Cryptography Tests
# Testing with QuantoniumOS crypto implementations
#
# ===================================================================

import unittest
import sys
import os
from binascii import unhexlify

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

# Import QuantoniumOS cryptography modules
try:
    from quantonium_crypto_production import QuantoniumCrypto
    from true_rft_feistel_bindings import TrueRFTFeistel
    from paper_compliant_crypto_bindings import PaperCompliantCrypto
except ImportError:
    # Fallback imports if modules are in different locations
    sys.path.insert(0, '/workspaces/quantoniumos/06_CRYPTOGRAPHY')
    from quantonium_crypto_production import QuantoniumCrypto
    from true_rft_feistel_bindings import TrueRFTFeistel

# Import QuantoniumOS RFT algorithms
try:
    sys.path.insert(0, '/workspaces/quantoniumos/04_RFT_ALGORITHMS')
    from canonical_true_rft import CanonicalTrueRFT
    from true_rft_exact import TrueRFTExact
    from true_rft_engine_bindings import TrueRFTEngineBindings
except ImportError:
    pass

# Import QuantoniumOS quantum engines
try:
    sys.path.insert(0, '/workspaces/quantoniumos/05_QUANTUM_ENGINES')
    from bulletproof_quantum_kernel import BulletproofQuantumKernel
    from topological_quantum_kernel import TopologicalQuantumKernel
    from working_quantum_kernel import WorkingQuantumKernel
except ImportError:
    pass

"""Self-test suite for Crypto.Hash.SHA224"""

# Test vectors from various sources
# This is a list of (expected_result, input[, description]) tuples.
test_data = [

    # RFC 3874: Section 3.1, "Test Vector #1 ('23097d223405d8228642a477bda255b32aadbce4bda0b3f7e36c9da7', 'abc'), # RFC 3874: Section 3.2, "Test Vector #2
    ('75388b16512776cc5dba5da1fd890150b0c6455cb4f58b1952522525', 'abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq'),

    # RFC 3874: Section 3.3, "Test Vector #3 ('20794655980c91d8bbb4c1ea97618a4bf03f42581948b2ee4ee7ad67', 'a' * 10**6, "'a' * 10**6"), # Examples from http://de.wikipedia.org/wiki/Secure_Hash_Algorithm ('d14a028c2a3a2bc9476102bb288234c415a2b01f828ea62ac5b3e42f', ''), ('49b08defa65e644cbf8a2dd9270bdededabc741997d1dadd42026d7b', 'Franz jagt im komplett verwahrlosten Taxi quer durch Bayern'), ('58911e7fccf2971a7d07f93162d8bd13568e71aa8fc86fc1fe9043d1', 'Frank jagt im komplett verwahrlosten Taxi quer durch Bayern'), ] def get_tests(config={}): from Crypto.Hash import SHA224 from .common
import make_hash_tests return make_hash_tests(SHA224, "SHA224", test_data,
        digest_size=28,
        oid='2.16.840.1.101.3.4.2.4')

if __name__ == '__main__':
    import unittest
    suite = lambda: unittest.TestSuite(get_tests())
    unittest.main(defaultTest='suite')

# vim:set ts=4 sw=4 sts=4 expandtab:
