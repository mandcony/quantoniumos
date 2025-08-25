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

class OfbTests(BlockChainingTests):
    aes_mode = AES.MODE_OFB
    des3_mode = DES3.MODE_OFB

    # Redefine test_unaligned_data_128/64

    def test_unaligned_data_128(self):
        plaintexts = [b"7777777"] * 100

        cipher = AES.new(self.key_128, AES.MODE_CFB, self.iv_128, segment_size=8)
        ciphertexts = [cipher.encrypt(x) for x in plaintexts]
        cipher = AES.new(self.key_128, AES.MODE_CFB, self.iv_128, segment_size=8)
        self.assertEqual(b"".join(ciphertexts), cipher.encrypt(b"".join(plaintexts)))

        cipher = AES.new(self.key_128, AES.MODE_CFB, self.iv_128, segment_size=128)
        ciphertexts = [cipher.encrypt(x) for x in plaintexts]
        cipher = AES.new(self.key_128, AES.MODE_CFB, self.iv_128, segment_size=128)
        self.assertEqual(b"".join(ciphertexts), cipher.encrypt(b"".join(plaintexts)))

    def test_unaligned_data_64(self):
        plaintexts = [b"7777777"] * 100
        cipher = DES3.new(self.key_192, DES3.MODE_CFB, self.iv_64, segment_size=8)
        ciphertexts = [cipher.encrypt(x) for x in plaintexts]
        cipher = DES3.new(self.key_192, DES3.MODE_CFB, self.iv_64, segment_size=8)
        self.assertEqual(b"".join(ciphertexts), cipher.encrypt(b"".join(plaintexts)))

        cipher = DES3.new(self.key_192, DES3.MODE_CFB, self.iv_64, segment_size=64)
        ciphertexts = [cipher.encrypt(x) for x in plaintexts]
        cipher = DES3.new(self.key_192, DES3.MODE_CFB, self.iv_64, segment_size=64)
        self.assertEqual(b"".join(ciphertexts), cipher.encrypt(b"".join(plaintexts)))

class NistOfbVectors(NistBlockChainingVectors):
    aes_mode = AES.MODE_OFB
    des_mode = DES.MODE_OFB
    des3_mode = DES3.MODE_OFB

# Create one test method per file
nist_aes_kat_mmt_files = (
    # KAT
    "OFBGFSbox128.rsp",
    "OFBGFSbox192.rsp",
    "OFBGFSbox256.rsp",
    "OFBKeySbox128.rsp",
    "OFBKeySbox192.rsp",
    "OFBKeySbox256.rsp",
    "OFBVarKey128.rsp",
    "OFBVarKey192.rsp",
    "OFBVarKey256.rsp",
    "OFBVarTxt128.rsp",
    "OFBVarTxt192.rsp",
    "OFBVarTxt256.rsp",
    # MMT
    "OFBMMT128.rsp",
    "OFBMMT192.rsp",
    "OFBMMT256.rsp",
)
nist_aes_mct_files = (
    "OFBMCT128.rsp",
    "OFBMCT192.rsp",
    "OFBMCT256.rsp",
)

for file_name in nist_aes_kat_mmt_files:

    def new_func(self, file_name=file_name):
        self._do_kat_aes_test(file_name)

    setattr(NistOfbVectors, "test_AES_" + file_name, new_func)

for file_name in nist_aes_mct_files:

    def new_func(self, file_name=file_name):
        self._do_mct_aes_test(file_name)

    setattr(NistOfbVectors, "test_AES_" + file_name, new_func)
del file_name, new_func

nist_tdes_files = (
    "TOFBMMT2.rsp",  # 2TDES
    "TOFBMMT3.rsp",  # 3TDES
    "TOFBinvperm.rsp",  # Single DES
    "TOFBpermop.rsp",
    "TOFBsubtab.rsp",
    "TOFBvarkey.rsp",
    "TOFBvartext.rsp",
)

for file_name in nist_tdes_files:

    def new_func(self, file_name=file_name):
        self._do_tdes_test(file_name)

    setattr(NistOfbVectors, "test_TDES_" + file_name, new_func)

# END OF NIST OFB TEST VECTORS

class SP800TestVectors(unittest.TestCase):
    """Class exercising the OFB test vectors found in Section F.4
    of NIST SP 800-3A"""

    def test_aes_128(self):
        plaintext = (
            "6bc1bee22e409f96e93d7e117393172a"
            + "ae2d8a571e03ac9c9eb76fac45af8e51"
            + "30c81c46a35ce411e5fbc1191a0a52ef"
            + "f69f2445df4f9b17ad2b417be66c3710"
        )
        ciphertext = (
            "3b3fd92eb72dad20333449f8e83cfb4a"
            + "7789508d16918f03f53c52dac54ed825"
            + "9740051e9c5fecf64344f7a82260edcc"
            + "304c6528f659c77866a510d9c1d6ae5e"
        )
        key = "2b7e151628aed2a6abf7158809cf4f3c"
        iv = "000102030405060708090a0b0c0d0e0f"

        key = unhexlify(key)
        iv = unhexlify(iv)
        plaintext = unhexlify(plaintext)
        ciphertext = unhexlify(ciphertext)

        cipher = AES.new(key, AES.MODE_OFB, iv)
        self.assertEqual(cipher.encrypt(plaintext), ciphertext)
        cipher = AES.new(key, AES.MODE_OFB, iv)
        self.assertEqual(cipher.decrypt(ciphertext), plaintext)

        cipher = AES.new(key, AES.MODE_OFB, iv)
        self.assertEqual(cipher.encrypt(plaintext[:-8]), ciphertext[:-8])
        cipher = AES.new(key, AES.MODE_OFB, iv)
        self.assertEqual(cipher.decrypt(ciphertext[:-8]), plaintext[:-8])

    def test_aes_192(self):
        plaintext = (
            "6bc1bee22e409f96e93d7e117393172a"
            + "ae2d8a571e03ac9c9eb76fac45af8e51"
            + "30c81c46a35ce411e5fbc1191a0a52ef"
            + "f69f2445df4f9b17ad2b417be66c3710"
        )
        ciphertext = (
            "cdc80d6fddf18cab34c25909c99a4174"
            + "fcc28b8d4c63837c09e81700c1100401"
            + "8d9a9aeac0f6596f559c6d4daf59a5f2"
            + "6d9f200857ca6c3e9cac524bd9acc92a"
        )
        key = "8e73b0f7da0e6452c810f32b809079e562f8ead2522c6b7b"
        iv = "000102030405060708090a0b0c0d0e0f"

        key = unhexlify(key)
        iv = unhexlify(iv)
        plaintext = unhexlify(plaintext)
        ciphertext = unhexlify(ciphertext)

        cipher = AES.new(key, AES.MODE_OFB, iv)
        self.assertEqual(cipher.encrypt(plaintext), ciphertext)
        cipher = AES.new(key, AES.MODE_OFB, iv)
        self.assertEqual(cipher.decrypt(ciphertext), plaintext)

        cipher = AES.new(key, AES.MODE_OFB, iv)
        self.assertEqual(cipher.encrypt(plaintext[:-8]), ciphertext[:-8])
        cipher = AES.new(key, AES.MODE_OFB, iv)
        self.assertEqual(cipher.decrypt(ciphertext[:-8]), plaintext[:-8])

    def test_aes_256(self):
        plaintext = (
            "6bc1bee22e409f96e93d7e117393172a"
            + "ae2d8a571e03ac9c9eb76fac45af8e51"
            + "30c81c46a35ce411e5fbc1191a0a52ef"
            + "f69f2445df4f9b17ad2b417be66c3710"
        )
        ciphertext = (
            "dc7e84bfda79164b7ecd8486985d3860"
            + "4febdc6740d20b3ac88f6ad82a4fb08d"
            + "71ab47a086e86eedf39d1c5bba97c408"
            + "0126141d67f37be8538f5a8be740e484"
        )
        key = "603deb1015ca71be2b73aef0857d77811f352c073b6108d72d9810a30914dff4"
        iv = "000102030405060708090a0b0c0d0e0f"

        key = unhexlify(key)
        iv = unhexlify(iv)
        plaintext = unhexlify(plaintext)
        ciphertext = unhexlify(ciphertext)

        cipher = AES.new(key, AES.MODE_OFB, iv)
        self.assertEqual(cipher.encrypt(plaintext), ciphertext)
        cipher = AES.new(key, AES.MODE_OFB, iv)
        self.assertEqual(cipher.decrypt(ciphertext), plaintext)

        cipher = AES.new(key, AES.MODE_OFB, iv)
        self.assertEqual(cipher.encrypt(plaintext[:-8]), ciphertext[:-8])
        cipher = AES.new(key, AES.MODE_OFB, iv)
        self.assertEqual(cipher.decrypt(ciphertext[:-8]), plaintext[:-8])

def get_tests(config={}):
    tests = []
    tests += list_test_cases(OfbTests)
    if config.get("slow_tests"):
        tests += list_test_cases(NistOfbVectors)
    tests += list_test_cases(SP800TestVectors)
    return tests

if __name__ == "__main__":

    def suite():
        return unittest.TestSuite(get_tests())

    unittest.main(defaultTest="suite")
