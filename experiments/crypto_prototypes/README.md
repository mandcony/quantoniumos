# ⚠️ Experimental Cryptographic Prototypes

> **WARNING: RESEARCH ONLY — NOT FOR PRODUCTION USE**

This directory contains **experimental cryptographic implementations** that are:

1. **NOT standards-compliant** — Does not implement NIST PQC finalists (ML-KEM, ML-DSA)
2. **NOT audited** — No formal security review or cryptanalysis
3. **NOT constant-time** — May be vulnerable to timing side-channels
4. **NOT production-ready** — Missing DRBG, proper entropy sources, etc.

---

## Contents

| File | Description | Security Level |
|------|-------------|----------------|
| `enhanced_cipher.py` | 48-round Feistel with RFT-SIS hashes | ⚠️ Unverified |
| `primitives/` | Low-level RFT-based crypto primitives | ⚠️ Experimental |
| `benchmarks/` | Performance and avalanche analysis | N/A (test code) |

---

## Known Gaps vs NIST PQC Standards

| Requirement | NIST ML-KEM/ML-DSA | This Implementation |
|-------------|-------------------|---------------------|
| **Security Reduction** | Proven hardness (MLWE/MLSIS) | ❌ No formal proof |
| **Key Sizes** | 800–1568 bytes | ~3 KB (non-standard) |
| **Ciphertext Sizes** | 768–1568 bytes | ~4 KB (non-standard) |
| **Side-Channel Resistance** | Required | ❌ Not implemented |
| **DRBG** | NIST SP 800-90A | ❌ Uses numpy.random |
| **Test Vectors** | Official KATs | ❌ None |

---

## Intended Use

These prototypes are intended **exclusively** for:

- 📚 Academic research on transform-based cryptography
- 🧪 Exploring RFT properties in the cryptographic domain  
- 📊 Benchmarking avalanche effect and diffusion properties
- 🎓 Educational demonstrations

---

## DO NOT USE FOR

- ❌ Protecting real secrets or sensitive data
- ❌ Production systems or deployed applications
- ❌ Financial or medical data encryption
- ❌ Any scenario requiring actual security guarantees

---

## Migration Path to Standards-Compliant Crypto

If you need real cryptography, use:

```python
# For encryption (NIST ML-KEM / Kyber)
from liboqs import KeyEncapsulation
kem = KeyEncapsulation("Kyber768")

# For signatures (NIST ML-DSA / Dilithium)  
from liboqs import Signature
sig = Signature("Dilithium3")

# For symmetric encryption (FIPS 140-3)
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.backends import default_backend
```

---

## References

- [NIST Post-Quantum Cryptography](https://csrc.nist.gov/projects/post-quantum-cryptography)
- [CRYSTALS-Kyber](https://pq-crystals.org/kyber/)
- [CRYSTALS-Dilithium](https://pq-crystals.org/dilithium/)
- [Open Quantum Safe (liboqs)](https://github.com/open-quantum-safe/liboqs)

---

*Last updated: 2026-01-27*
