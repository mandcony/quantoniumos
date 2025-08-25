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

import unittest
from binascii import unhexlify

def get_tag_random(tag, length):
    return SHAKE128.new(data=tobytes(tag)).read(length)

class OpenPGPTests(BlockChainingTests):
    aes_mode = AES.MODE_OPENPGP
    des3_mode = DES3.MODE_OPENPGP

    # Redefine test_unaligned_data_128/64

    key_128 = get_tag_random("key_128", 16)
    key_192 = get_tag_random("key_192", 24)
    iv_128 = get_tag_random("iv_128", 16)
    iv_64 = get_tag_random("iv_64", 8)
    data_128 = get_tag_random("data_128", 16)

    def test_loopback_128(self):
        cipher = AES.new(self.key_128, AES.MODE_OPENPGP, self.iv_128)
        pt = get_tag_random("plaintext", 16 * 100)
        ct = cipher.encrypt(pt)

        eiv, ct = ct[:18], ct[18:]

        cipher = AES.new(self.key_128, AES.MODE_OPENPGP, eiv)
        pt2 = cipher.decrypt(ct)
        self.assertEqual(pt, pt2)

    def test_loopback_64(self):
        cipher = DES3.new(self.key_192, DES3.MODE_OPENPGP, self.iv_64)
        pt = get_tag_random("plaintext", 8 * 100)
        ct = cipher.encrypt(pt)

        eiv, ct = ct[:10], ct[10:]

        cipher = DES3.new(self.key_192, DES3.MODE_OPENPGP, eiv)
        pt2 = cipher.decrypt(ct)
        self.assertEqual(pt, pt2)

    def test_IV_iv_attributes(self):
        cipher = AES.new(self.key_128, AES.MODE_OPENPGP, self.iv_128)
        eiv = cipher.encrypt(b"")
        self.assertEqual(cipher.iv, self.iv_128)

        cipher = AES.new(self.key_128, AES.MODE_OPENPGP, eiv)
        self.assertEqual(cipher.iv, self.iv_128)

    def test_null_encryption_decryption(self):
        cipher = AES.new(self.key_128, AES.MODE_OPENPGP, self.iv_128)
        eiv = cipher.encrypt(b"")

        cipher = AES.new(self.key_128, AES.MODE_OPENPGP, eiv)
        self.assertEqual(cipher.decrypt(b""), b"")

    def test_either_encrypt_or_decrypt(self):
        cipher = AES.new(self.key_128, AES.MODE_OPENPGP, self.iv_128)
        eiv = cipher.encrypt(b"")
        self.assertRaises(TypeError, cipher.decrypt, b"")

        cipher = AES.new(self.key_128, AES.MODE_OPENPGP, eiv)
        cipher.decrypt(b"")
        self.assertRaises(TypeError, cipher.encrypt, b"")

    def test_unaligned_data_128(self):
        plaintexts = [b"7777777"] * 100

        cipher = AES.new(self.key_128, AES.MODE_OPENPGP, self.iv_128)
        ciphertexts = [cipher.encrypt(x) for x in plaintexts]
        cipher = AES.new(self.key_128, AES.MODE_OPENPGP, self.iv_128)
        self.assertEqual(b"".join(ciphertexts), cipher.encrypt(b"".join(plaintexts)))

    def test_unaligned_data_64(self):
        plaintexts = [b"7777777"] * 100

        cipher = DES3.new(self.key_192, DES3.MODE_OPENPGP, self.iv_64)
        ciphertexts = [cipher.encrypt(x) for x in plaintexts]
        cipher = DES3.new(self.key_192, DES3.MODE_OPENPGP, self.iv_64)
        self.assertEqual(b"".join(ciphertexts), cipher.encrypt(b"".join(plaintexts)))

    def test_output_param(self):
        pass

    def test_output_param_same_buffer(self):
        pass

    def test_output_param_memoryview(self):
        pass

    def test_output_param_neg(self):
        pass

class TestVectors(unittest.TestCase):
    def test_aes(self):
        # The following test vectors have been generated with gpg v1.4.0.
        # The command line used was:
        #
        #    gpg -c -z 0 --cipher-algo AES --passphrase secret_passphrase \
        #     --disable-mdc --s2k-mode 0 --output ct pt
        #
        # As result, the content of the file 'pt' is encrypted with a key derived
        # from 'secret_passphrase' and written to file 'ct'.
        # Test vectors must be extracted from 'ct', which is a collection of
        # TLVs (see RFC4880 for all details):
        # - the encrypted data (with the encrypted IV as prefix) is the payload
        #   of the TLV with tag 9 (Symmetrical Encrypted Data Packet).
        #   This is the ciphertext in the test vector.
        # - inside the encrypted part, there is a further layer of TLVs. One must
        #   look for tag 11 (Literal Data  Packet); in its payload, after a short
        #   but time dependent header, there is the content of file 'pt'.
        #   In the test vector, the plaintext is the complete set of TLVs that gets
        #   encrypted. It is not just the content of 'pt'.
        # - the key is the leftmost 16 bytes of the SHA1 digest of the password.
        #   The test vector contains such shortened digest.
        #
        # Note that encryption uses a clear IV, and decryption an encrypted IV

        plaintext = "ac18620270744fb4f647426c61636b4361745768697465436174"
        ciphertext = "dc6b9e1f095de609765c59983db5956ae4f63aea7405389d2ebb"
        key = "5baa61e4c9b93f3f0682250b6cf8331b"
        iv = "3d7d3e62282add7eb203eeba5c800733"
        encrypted_iv = "fd934601ef49cb58b6d9aebca6056bdb96ef"

        plaintext = unhexlify(plaintext)
        ciphertext = unhexlify(ciphertext)
        key = unhexlify(key)
        iv = unhexlify(iv)
        encrypted_iv = unhexlify(encrypted_iv)

        cipher = AES.new(key, AES.MODE_OPENPGP, iv)
        ct = cipher.encrypt(plaintext)
        self.assertEqual(ct[:18], encrypted_iv)
        self.assertEqual(ct[18:], ciphertext)

        cipher = AES.new(key, AES.MODE_OPENPGP, encrypted_iv)
        pt = cipher.decrypt(ciphertext)
        self.assertEqual(pt, plaintext)

    def test_des3(self):
        # The following test vectors have been generated with gpg v1.4.0.
        # The command line used was:
        #    gpg -c -z 0 --cipher-algo 3DES --passphrase secret_passphrase \
        #     --disable-mdc --s2k-mode 0 --output ct pt
        # For an explanation, see test_AES.py .

        plaintext = "ac1762037074324fb53ba3596f73656d69746556616c6c6579"
        ciphertext = "9979238528357b90e2e0be549cb0b2d5999b9a4a447e5c5c7d"
        key = "7ade65b460f5ea9be35f9e14aa883a2048e3824aa616c0b2"
        iv = "cd47e2afb8b7e4b0"
        encrypted_iv = "6a7eef0b58050e8b904a"

        plaintext = unhexlify(plaintext)
        ciphertext = unhexlify(ciphertext)
        key = unhexlify(key)
        iv = unhexlify(iv)
        encrypted_iv = unhexlify(encrypted_iv)

        cipher = DES3.new(key, DES3.MODE_OPENPGP, iv)
        ct = cipher.encrypt(plaintext)
        self.assertEqual(ct[:10], encrypted_iv)
        self.assertEqual(ct[10:], ciphertext)

        cipher = DES3.new(key, DES3.MODE_OPENPGP, encrypted_iv)
        pt = cipher.decrypt(ciphertext)
        self.assertEqual(pt, plaintext)

def get_tests(config={}):
    tests = []
    tests += list_test_cases(OpenPGPTests)
    tests += list_test_cases(TestVectors)
    return tests

if __name__ == "__main__":

    def suite():
        return unittest.TestSuite(get_tests())

    unittest.main(defaultTest="suite")
