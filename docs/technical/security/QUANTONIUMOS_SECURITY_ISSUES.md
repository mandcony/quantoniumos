# QuantoniumOS Security Issues Catalog

This document consolidates all currently known security findings for the QuantoniumOS resonance computing stack. Issues are grouped by severity and cross-reference the relevant components and source files so remediation work can be prioritized.

## High Severity

### H-1: Provenance Forgery
- **Component:** `GeometricWaveformHash`
- **Description:** Spectral fingerprints can be reproduced by any party because the hash pipeline relies entirely on public φ-derived constants with no secret salt.
- **Attack:** Recompute the geometric waveform hash over a poisoned model to forge provenance.
- **Impact:** Legitimate and malicious model updates are indistinguishable.
- **Locations:** `algorithms/rft/core/geometric_waveform_hash.py` (L53-L74, L150-L171)
- **Remediation:** Introduce secret or signer-bound salt (e.g., keyed HMAC or digital signature).

## Medium Severity

### M-1: Metadata Leakage
- **Component:** `Vertex Codec`
- **Description:** Manifests emit pruning thresholds, quantization bit-widths, tensor counts, and tensor names as plaintext JSON.
- **Attack:** Passive observers learn model structure without decrypting the payload.
- **Impact:** Facilitates targeted attacks and reveals proprietary model details.
- **Location:** `algorithms/compression/vertex/rft_vertex_codec.py` (L823-L874)
- **Remediation:** Encrypt manifests before transport or strip metadata for untrusted recipients.

### M-2: Shared RFT Basis Coupling
- **Components:** Codec + Hash + Crypto integration
- **Description:** All stages share the same φ-weighted basis and low cipher diffusion leaves resonance coefficients partially correlated across stages.
- **Attack:** Multi-stage side-channel combining leaked tensors and ciphertext biases to recover plaintext structure.
- **Impact:** Enables partial plaintext reconstruction from ciphertext statistics.
- **Locations:**
  - `algorithms/rft/core/enhanced_rft_crypto_v2.py` (L348-L414, L604-L674)
  - `algorithms/compression/vertex/rft_vertex_codec.py` (L823-L874)
- **Remediation:** Improve cipher diffusion (fix avalanche) and randomize codec basis per deployment.

### M-3: Low Avalanche Effect (0.438)
- **Component:** `EnhancedRFTCryptoV2`
- **Description:** Message avalanche measures 0.438, below the ≥0.48 cryptographic threshold.
- **Attack:** Correlate ciphertext bit flips with plaintext variations, especially across related model checkpoints.
- **Impact:** Enables statistical inference of plaintext structure.
- **Location:** `algorithms/rft/core/enhanced_rft_crypto_v2.py` (L604-L674)
- **Remediation:** Increase rounds, strengthen `MixColumns`, and add nonlinearity.

### M-4: Deterministic Manifold Seeding
- **Component:** `GeometricWaveformHash`
- **Description:** Manifold matrices are seeded from public φ-derived values, making spectral signatures reproducible.
- **Attack:** Recreate the manifold to forge signatures or mount oracle attacks.
- **Impact:** No secrecy in hash generation; enables provenance forgery.
- **Location:** `algorithms/rft/core/geometric_waveform_hash.py` (L53-L74)
- **Remediation:** Use deployment-specific secret salts for RFT basis generation.

### M-5: φ-Space Collision Risk
- **Component:** `GeometricWaveformHash`
- **Description:** Fixed low-dimensional manifold with coarse quantization permits near-collisions in φ-space.
- **Attack:** Search for inputs that map to similar manifold coordinates, then collide after rounding.
- **Impact:** Facilitates forging similar spectral fingerprints despite SHA-256 resistance.
- **Location:** `algorithms/rft/core/geometric_waveform_hash.py` (L115-L127)
- **Remediation:** Increase manifold dimensionality or raise quantization precision.

## Low Severity

### L-1: Boot Integrity Not Verified
- **Component:** `quantonium_boot.py`
- **Description:** `validate_core_algorithms` only checks file existence.
- **Attack:** Replace core modules with backdoored versions without detection.
- **Impact:** Allows compromised releases to load silently.
- **Location:** `quantonium_boot.py` (L146-L166)
- **Remediation:** Verify signed digests before loading core components.

### L-2: No End-to-End Signature Chain
- **Component:** Workflow integration
- **Description:** No signature binds codec output, hash, and ciphertext to a release identity.
- **Attack:** Man-in-the-middle can swap artifacts between stages.
- **Impact:** Users consume tampered data without detection.
- **Location:** `docs/technical/guides/RESONANCE_COMPUTING_PLAYBOOK.md` (L13-L31)
- **Remediation:** Add a digital signature covering the full workflow chain.

## Informational

### I-1: Router Replay Vulnerability
- **Component:** `CompressedModelRouter`
- **Description:** Router trusts any sealed artifact without checking freshness.
- **Attack:** Replay stale but valid artifacts to downgrade users.
- **Impact:** Users unknowingly run outdated or vulnerable models.
- **Location:** `quantonium_os_src/apps/system/compressed_model_router.py` (L120-L195)
- **Remediation:** Track publication epochs and reject reused salts/versions.

### I-2: No Manifest Authentication
- **Component:** `CompressedModelRouter`
- **Description:** Router enumerates manifests without verifying authenticity.
- **Attack:** Inject malicious manifests when filesystem access is available.
- **Impact:** Malicious models appear legitimate.
- **Location:** `quantonium_os_src/apps/system/compressed_model_router.py` (L120-L195)
- **Remediation:** Require manifest signatures before indexing.

## Architectural / Design Issues

### A-1: Deterministic RFT Matrix
- **Component:** `CanonicalTrueRFT`
- **Description:** Transform matrix Ψ is deterministic for a given size.
- **Attack:** Precompute Ψ to craft poisoning payloads or weaken diffusion.
- **Location:** `algorithms/rft/core/geometric_waveform_hash.py` (L77-L171)
- **Remediation:** Randomize RFT basis per deployment and keep seeds confidential.

### A-2: Linear Transform Vulnerability
- **Component:** RFT core
- **Description:** Strict linearity (Ψ(x + y) = Ψx + Ψy) simplifies algebraic attacks.
- **Remediation:** Introduce nonlinear operations or limit RFT use to non-security-critical stages.

### A-3: Shared Spectral Coordinates
- **Component:** System architecture
- **Description:** Compression, hashing, and encryption reuse the same spectral space, coupling failures.
- **Remediation:** Separate bases for security-critical paths or provide composition proofs.

### A-4: Public Algorithm Parameters
- **Component:** Entire stack
- **Description:** φ sequences, Gaussian weights, and QR processes are public.
- **Impact:** Aligns with Kerckhoffs's principle but requires robust key management.

## Information Flow Issues

- **IF-1: Codec Metadata Exposure:** Metadata remains in cleartext; see M-1.
- **IF-2: Spectral Signature Reproducibility:** Deterministic manifolds allow signature cloning; see H-1/M-4.
- **IF-3: Ciphertext-Plaintext Correlation:** Low diffusion correlates ciphertext and plaintext resonances; see M-2/M-3.

## Demonstrated Attack Scenarios

### AS-1: Model Poisoning Attack (Feasibility: High)
1. Gain write access to distribution.
2. Swap codec output with poisoned tensors.
3. Recompute geometric waveform hash (deterministic) and re-encrypt with attacker key.
4. Router accepts artifact without signature verification.

### AS-2: Spectral Side-Channel Attack (Feasibility: Medium)
1. Observe ciphertexts for sequential checkpoints.
2. Exploit low avalanche to correlate bit flips.
3. Use leaked codec metadata to constrain hypotheses.
4. Recover coarse architecture details.

### AS-3: Resonance Oracle Attack (Feasibility: Medium-High)
1. Query public interfaces with crafted payloads.
2. Observe manifold outputs or ciphertext biases.
3. Solve for RFT columns via `_bytes_to_signal` leakage.
4. Craft optimized poisoning payloads.

## Remediation Roadmap

| Priority | Issues | Estimated Effort |
|----------|--------|------------------|
| **P0** | H-1 | ~1 week |
| **P1** | M-1, M-2, M-3, M-4, M-5 | 7–12 weeks |
| **P2** | L-1, L-2, I-1, I-2 | 4 weeks |
| **Architectural** | A-1 – A-4 | Ongoing |

Total remediation timeline is estimated at approximately four months when addressed sequentially.

