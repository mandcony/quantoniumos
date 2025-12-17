# /hardware/

## ARCHITECTURAL FEASIBILITY STUDY

Mandatory labels:
- **No silicon fabricated**
- **Performance based on simulation/synthesis only**
- **Not a claim of optimality**

This tree contains RTL/TL-Verilog/FPGA builds, testbenches, synthesis reports, and visualization tooling for the RFTPU.

### 🧬 Verified Configuration
The `fpga_top.json` file defines the 16-mode configuration used for WebFPGA synthesis. This configuration has been updated to reflect the **Verified Scientific Findings** (Dec 2025):

- **Mode 0 (Golden):** Verified Class B Transform.
- **Mode 6 (Cascade):** Verified Class B Compression Winner (H3).
- **Mode 12 (SIS Hash):** Verified Class D Crypto.
- **Mode 14 (Quantum):** Verified Class A Symbolic Engine.

### Subsections
- `hardware/rtl/` — RTL sources and architecture definitions (indexed; files may still live at repo root `hardware/` for compatibility)
- `hardware/simulation/` — testbenches and simulation harnesses (indexed)
- `hardware/architecture_docs/` — design docs, reports, and figures (indexed)
