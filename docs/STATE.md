# QuantoniumOS Operational State

**Status:** Active Research
**Current Session:** [2025-12-17](sessions/2025-12-17_SESSION.md)

---

## 1. Active Goals (Max 5)

1.  **Establish Operational Rigor:** Implement and enforce "Two-File Rule" (STATE + SESSION) to bind intent to artifacts.
2.  **Formalize Canonical RFT:** Solidify the "Gram-normalized irrational-frequency exponential basis" as the single source of truth for RFT v2.0.0.
3.  **Release Management:** Maintain v2.0.0 stability and documentation alignment.
4.  **Hardware Validation:** Verify RFTPU RTL matches the new Canonical RFT definition.

---

## 2. Claims Matrix

| Claim | Status | Confidence | Evidence / Artifact |
| :--- | :--- | :--- | :--- |
| **Canonical RFT Unitarity** | ✅ **INTERNALLY PROVEN** | High | `test_phi_frame_normalization.py`, Theorem 2.1 (Gram normalization) |
| **Sparsity Advantage** | ✅ **INTERNALLY PROVEN** | High | +15-20dB on golden-ratio signals (Benchmarks) |
| **Hardware Feasibility (FPGA)** | ✅ **INTERNALLY PROVEN** | High | RTL Synthesis (`fpga_top.sv`), WebFPGA validation |
| **General Superiority** | 🧪 **EXPERIMENTAL** | Low | No advantage on white noise; specific to quasi-periodic domain |
| **Crypto Security** | 🧪 **EXPERIMENTAL** | Low | Avalanche metrics observed; no formal reduction to hard problems |
| **Quantum Simulation** | 🧪 **EXPERIMENTAL** | Medium | Classical simulation of wave-domain logic (505 Mq/s claimed) |

---

## 3. Open Questions

1.  **LCT Conjecture:** Is the Canonical RFT structurally distinct from the Linear Canonical Transform (LCT) group? (Status: OPEN)
2.  **Structural Distinctness:** Is RFT distinct from the DFT orbit for all $N$? (Proven for $N \le 32$, Open for general $N$)
3.  **Large-N Scalability:** Gram normalization is $O(N^3)$. Can we achieve $O(N \log N)$ unitarity for $N > 4096$?

---

## 4. Hard Constraints

*   **Patent:** USPTO Application #19/169,399 (Filed 2025-04-03).
*   **Licensing:**
    *   Core: AGPL-3.0-or-later.
    *   Claims-Practicing Files: `LICENSE-CLAIMS-NC.md` (Non-Commercial / Research Only).
*   **Mode:** Research & Education. Not suitable for production cryptographic use. No security guarantees claimed.
