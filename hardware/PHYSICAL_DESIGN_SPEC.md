# QuantoniumOS RPU Physical Design Specification

**Derived from RTL Analysis** — December 2025  
**Target**: ASIC tape-out ready specification (TSMC N7 reference)

---

## 1. Process & Technology Stack

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| **Logic Node** | TSMC N7FF (7nm FinFET) | Optimal power/perf for DSP+crypto workloads |
| **Metal Stack** | 12M (8 thin + 3 thick + 1 RDL) | RFT MAC arrays need dense local routing; thick metals for power grid |
| **Packaging** | 2.5D CoWoS-S interposer | Allows HBM2E integration + multi-die expansion |
| **TSV Pitch** | 40 µm (keep-out 20 µm) | Standard for TSMC CoWoS |
| **Voltage Domains** | 0.75V (core) / 1.8V (IO) / 1.1V (SRAM) | Multi-Vt library for power gating |

### Die Stack (Bottom to Top)
1. **Silicon Interposer** (passive) — 100 µm thick, 65 nm RDL
2. **Compute Die** — 8×8 tile array + controllers (this spec)
3. **HBM2E Stack** (2× chiplets, 8 GB each) — bonded to interposer
4. **Package Substrate** — 12-layer organic, 0.8 mm pitch BGA

---

## 2. Module Hierarchy & Area Estimates

Derived from your RTL (`quantoniumos_unified_engines.sv`, `rft_middleware_engine.sv`, `rpu_architecture.tlv`):

| Module | Instances | Gate Count (est.) | Area (µm²) @ N7 | Notes |
|--------|-----------|-------------------|-----------------|-------|
| `cordic_sincos` | 64 | 4,200 | 1,680 | 16-iter CORDIC per tile |
| `cordic_cartesian_to_polar` | 64 | 3,800 | 1,520 | 12-iter variant for polar extraction |
| `complex_mult` | 64 | 2,100 | 840 | 4 DSP mults per instance |
| `rft_kernel_rom` | 64 | 1,024 | 410 | 8×8 Q1.15 coefficient LUT |
| `rft_middleware_engine` | 64 | 18,500 | 7,400 | Complete 8-point RFT pipeline |
| `phi_rft_core` | 64 | 24,000 | 9,600 | 8-sample Φ-RFT + SIS digest |
| `canonical_rft_core` (N=64) | 1 | 185,000 | 74,000 | Large block for SIS hash path |
| `rft_sis_hash_v31` | 1 | 320,000 | 128,000 | N=512 expansion + lattice math |
| `feistel_round_function` | 1 | 8,500 | 3,400 | Combinational F-function |
| `feistel_48_cipher` | 1 | 45,000 | 18,000 | 48-round controller |
| `rpu_tile_shell` | 64 | 32,000 | 12,800 | FSM + scratch + topo mem |
| `rpu_dma_ingress` | 1 | 6,500 | 2,600 | 64-tile demux |
| `rpu_noc_fabric` | 1 | 95,000 | 38,000 | 64 routers, 64 inflight slots |
| `quantoniumos_unified_core` | 1 | 420,000 | 168,000 | Top controller + pipeline FSM |

### Summary
- **Total Logic Gates**: ~3.2 M gates
- **Total Logic Area**: ~1.28 mm²
- **SRAM Macros**: 128× 2Kb (scratch) + 64× 4Kb (topo) + 2× 256Kb (SIS A-matrix cache) = 640 Kb
- **SRAM Area**: ~0.45 mm²
- **Estimated Die Size (active)**: ~2.8 mm² (logic + SRAM + routing overhead 60%)

---

## 3. Power, Performance & Thermal Targets

### 3.1 Frequency Targets

| Clock Domain | Target Freq | Source | Consumers |
|--------------|-------------|--------|-----------|
| `clk_tile` | 950 MHz | PLL ÷4 | phi_rft_core, tile_shell, DMA ingress |
| `clk_noc` | 1.2 GHz | PLL ÷3.17 | rpu_noc_fabric (higher BW) |
| `clk_sis` | 475 MHz | PLL ÷8 | rft_sis_hash_v31 (power constrained) |
| `clk_feistel` | 1.4 GHz | PLL ÷2.7 | feistel_48_cipher (latency critical) |

### 3.2 Power Budget

| Block | Active (mW) | Idle (mW) | Notes |
|-------|-------------|-----------|-------|
| Single Tile (RFT+shell) | 85 | 4 | Power-gated SRAM |
| 64 Tiles total | 5,440 | 256 | Staggered activation |
| NoC Fabric | 620 | 45 | Clock-gated routers |
| SIS Hash Engine | 1,100 | 120 | Infrequent use |
| Feistel-48 | 280 | 15 | Stream cipher mode |
| DMA Ingress | 95 | 8 | Always-on path |
| Unified Controller | 180 | 12 | Minimal FSM |
| IO + PLL + Misc | 450 | 85 | Ring oscillators, pads |
| **Total** | **8,165** | **541** | Peak < 9W with headroom |

### 3.3 Thermal Constraints

- **Junction Temp Max**: 105°C (commercial)
- **Package θJA**: 8.5 °C/W (CoWoS with active cooling)
- **Hotspot Budget**: Keep any 0.5mm² region < 150 mW/mm² → enforced by staggered tile activation
- **Recommended**: Diagonal tile activation pattern (tiles 0,9,18,27… first, then 1,10,19…)

---

## 4. IO Budget & External Interfaces

### 4.1 Memory Channels (on interposer)

| Interface | Count | Width | Bandwidth | Purpose |
|-----------|-------|-------|-----------|---------|
| HBM2E PHY | 4 channels | 128-bit each | 460 GB/s total | DMA sample data |
| SRAM-to-interposer | 16 lanes | 64-bit | Local SIS cache |

### 4.2 Host & Control

| Interface | Pins | Standard | Purpose |
|-----------|------|----------|---------|
| PCIe Gen5 x8 | 16 diff pairs | CXL 2.0 ready | Host commands, config |
| JTAG | 5 | IEEE 1149.1 | Debug/scan |
| SPI | 4 | Quad SPI | Boot ROM, firmware |

### 4.3 Expansion & Debug

| Interface | Pins | Purpose |
|-----------|------|---------|
| Cascade Links | 4× LVDS pairs | Multi-die H3 fan-out |
| GPIO | 16 | Lab stimulus (matches Makerchip harness) |
| Analog Test | 8 | PLL lock, voltage monitors |

### 4.4 Pin Summary
- **Signal Pins**: ~420
- **Power/Ground**: ~380 (for IR drop < 3%)
- **Total**: ~800 pins (25×25 mm flip-chip)

---

## 5. Floorplan Constraints

### 5.1 Die Geometry

```
┌────────────────────────────────────────────────────────────────┐
│                           8.5 mm                               │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░│  │
│  │░  PLL_NW                                        PLL_SE  ░│  │  8.5 mm
│  │░                                                        ░│  │
│  │░    ┌────┬────┬────┬────┬────┬────┬────┬────┐          ░│  │
│  │░    │T00 │T01 │T02 │T03 │T04 │T05 │T06 │T07 │          ░│  │
│  │░    ├────┼────┼────┼────┼────┼────┼────┼────┤          ░│  │
│  │░    │T08 │T09 │T10 │T11 │T12 │T13 │T14 │T15 │          ░│  │
│  │░    ├────┼────┼────┼────┼────┼────┼────┼────┤          ░│  │
│  │░    │T16 │T17 │T18 │T19 │T20 │T21 │T22 │T23 │          ░│  │
│  │░    ├────┼────┼────┼────┼────┼────┼────┼────┤  SPINE   ░│  │
│  │░    │T24 │T25 │T26 │T27 │T28 │T29 │T30 │T31 │◄─SIS/Ctrl░│  │
│  │░    ├────┼────┼────┼────┼────┼────┼────┼────┤          ░│  │
│  │░    │T32 │T33 │T34 │T35 │T36 │T37 │T38 │T39 │          ░│  │
│  │░    ├────┼────┼────┼────┼────┼────┼────┼────┤          ░│  │
│  │░    │T40 │T41 │T42 │T43 │T44 │T45 │T46 │T47 │          ░│  │
│  │░    ├────┼────┼────┼────┼────┼────┼────┼────┤          ░│  │
│  │░    │T48 │T49 │T50 │T51 │T52 │T53 │T54 │T55 │          ░│  │
│  │░    ├────┼────┼────┼────┼────┼────┼────┼────┤          ░│  │
│  │░    │T56 │T57 │T58 │T59 │T60 │T61 │T62 │T63 │          ░│  │
│  │░    └────┴────┴────┴────┴────┴────┴────┴────┘          ░│  │
│  │░                                                        ░│  │
│  │░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░│  │
│  │            DMA INGRESS ROUTERS (South Edge)              │  │
│  └──────────────────────────────────────────────────────────┘  │
│                         IO RING                                │
└────────────────────────────────────────────────────────────────┘

Legend:
░ = Guard ring / ESD / TSV keep-out (400 µm)
SPINE = SIS hash engine + Feistel + unified controller
```

### 5.2 Block Placement Rules

| Block | Location | Aspect Ratio | Notes |
|-------|----------|--------------|-------|
| Tile (×64) | 6.5 mm × 6.5 mm central grid | 1:1 each ~0.75 mm | Hard macro |
| SIS Hash | x=7.2 mm, y=3.5–5.5 mm | 3:1 tall | Near center for NoC access |
| Feistel-48 | x=7.2 mm, y=2.0–3.5 mm | 2:1 | Below SIS |
| Unified Ctrl | x=7.2 mm, y=5.5–6.5 mm | 1:1 | Top of spine |
| DMA Ingress | y=0.5–1.0 mm, full width | 8:1 wide | Near HBM µbumps |
| NoC Routers | Embedded at tile edges | N/A | Soft cells |
| PLL_NW | x=0.5 mm, y=7.5 mm | 1:1 | Isolated analog island |
| PLL_SE | x=7.5 mm, y=0.5 mm | 1:1 | Isolated analog island |

### 5.3 Keep-Out Zones

- 400 µm perimeter for ESD ring + TSV landing
- 50 µm halo around PLL islands
- No logic under HBM µbump field (south 1.2 mm strip)

---

## 6. Clock Tree & Power Grid

### 6.1 Clock Distribution

```
                    ┌──────────┐
                    │  PLL_NW  │──────┐
                    └──────────┘      │
                                      ▼
                              ┌───────────────┐
          ┌───────────────────│  CLOCK SPINE  │───────────────────┐
          │                   │   (H-tree)    │                   │
          │                   └───────────────┘                   │
          ▼                          │                            ▼
    ┌──────────┐                     │                     ┌──────────┐
    │ Row 0-3  │◄────────────────────┼────────────────────►│ Row 4-7  │
    │ CTS Buf  │                     │                     │ CTS Buf  │
    └──────────┘                     │                     └──────────┘
          │                          │                            │
          ▼                          ▼                            ▼
    ┌───────────────────────────────────────────────────────────────┐
    │                    64 Tile Leaf Buffers                        │
    └───────────────────────────────────────────────────────────────┘
```

- **Skew Target**: < 50 ps across all tiles
- **Insertion Delay**: < 800 ps from PLL to leaf
- **OCV Margin**: 10% (process) + 5% (voltage) + 3% (temperature)

### 6.2 Power Grid

| Metal | Direction | Pitch | Width | Purpose |
|-------|-----------|-------|-------|---------|
| M12 (RDL) | X | 20 µm | 8 µm | VDD/VSS trunk |
| M11 | Y | 16 µm | 6 µm | VDD/VSS trunk |
| M10 | X | 12 µm | 4 µm | Domain VDD_SRAM |
| M9 | Y | 12 µm | 4 µm | Domain VDD_SRAM |
| M1-M8 | Alternating | 0.5-2 µm | 0.2-1 µm | Signal + local power taps |

- **IR Drop Target**: < 3% (22 mV at 0.75V core)
- **EM Limit**: J_avg < 1.5 mA/µm (DC), < 6 mA/µm (RMS AC)

---

## 7. Netlist-to-Macro Mapping

### 7.1 Hard Macros (pre-characterized)

| Macro | Count | Source | Liberty Model |
|-------|-------|--------|---------------|
| SRAM 2Kb (32×64) | 128 | TSMC memory compiler | `tsmc7ff_sram_2048x32_1rw.lib` |
| SRAM 4Kb (64×64) | 64 | TSMC memory compiler | `tsmc7ff_sram_4096x64_1rw.lib` |
| SRAM 256Kb | 2 | TSMC memory compiler | `tsmc7ff_sram_262144x32_1rw.lib` |
| PLL (3.8 GHz) | 2 | Analog IP | `tsmc7ff_pll_4ghz.lib` |
| HBM2E PHY | 4 | TSMC HBM IP | `tsmc7ff_hbm2e_phy.lib` |
| PCIe Gen5 PHY | 1 | Synopsys DesignWare | `dwc_pcie5_phy.lib` |

### 7.2 Soft Macros (synthesized)

| Module | Flatten? | Timing Constraint | Notes |
|--------|----------|-------------------|-------|
| `phi_rft_core` | No | 1.05 ns (950 MHz) | Hierarchical for ECO |
| `rpu_tile_shell` | No | 1.05 ns | Contains memories |
| `rpu_noc_fabric` | Partial | 0.83 ns (1.2 GHz) | Flatten router slices |
| `feistel_48_cipher` | Yes | 0.71 ns (1.4 GHz) | Fully flatten for timing |
| `rft_sis_hash_v31` | No | 2.1 ns (475 MHz) | Large, keep hierarchy |

---

## 8. Tool Flow

### 8.1 Front-End (RTL → Netlist)

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│ VCS + Verdi │ ──► │  Spyglass   │ ──► │   Genus     │
│  (Sim/Debug)│     │   (Lint)    │     │ (Synthesis) │
└─────────────┘     └─────────────┘     └─────────────┘
                                              │
                          Multi-mode multi-corner
                          (0.72V fast, 0.9V slow)
                                              ▼
                                     ┌─────────────┐
                                     │ Gate Netlist │
                                     │  (.v + .sdc) │
                                     └─────────────┘
```

### 8.2 Back-End (Netlist → GDSII)

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│   Innovus   │ ──► │  Tempus STA │ ──► │   Voltus    │
│  (P&R)      │     │  (Timing)   │     │ (Power/IR)  │
└─────────────┘     └─────────────┘     └─────────────┘
       │                                      │
       ▼                                      ▼
┌─────────────┐                      ┌─────────────┐
│   Calibre   │ ◄──────────────────► │  RedHawk    │
│  (DRC/LVS)  │                      │  (Thermal)  │
└─────────────┘                      └─────────────┘
```

### 8.3 Signoff Checklist

| Check | Tool | Pass Criteria |
|-------|------|---------------|
| Functional Sim | VCS | 100% test coverage |
| Lint | Spyglass | 0 errors, < 50 warnings |
| Synthesis QoR | Genus | WNS > 0, TNS = 0 |
| STA | Tempus | Setup/hold met all corners |
| IR Drop | Voltus | < 3% VDD |
| EM | Voltus | All nets pass J_avg limit |
| DRC | Calibre | 0 violations |
| LVS | Calibre | Clean match |
| Antenna | Calibre | 0 violations |
| ESD | Calibre | All pads protected |
| Thermal | RedHawk | Tj < 105°C |

---

## 9. Remaining Inputs Required

Before tape-out scripts can be finalized:

| Item | Status | Owner Action |
|------|--------|--------------|
| TSMC N7 PDK access | ❌ Needed | Obtain NDA + PDK |
| Memory compiler license | ❌ Needed | Request from TSMC |
| HBM2E PHY license | ❌ Needed | TSMC IP portal |
| PCIe/CXL PHY | ❌ Needed | Synopsys DesignWare license |
| Final tile count (48 vs 64) | ⚠️ Confirm | User decision |
| Power domain strategy | ⚠️ Confirm | Per-row vs per-tile |
| Package substrate vendor | ❌ Needed | ASE, Amkor, etc. |

---

## 10. Appendix: File Manifest

| File | Purpose |
|------|---------|
| `hardware/PHYSICAL_DESIGN_SPEC.md` | This document |
| `hardware/constraints/rpu_synthesis.sdc` | Synthesis timing constraints |
| `hardware/scripts/rpu_floorplan.tcl` | Innovus floorplan script |
| `tools/rpu_chip_visual_3d.py` | 3D visualization generator |
| `figures/rpu_chip_3d.png` | Rendered chip visualization |

---

*Document generated from RTL analysis of quantoniumos commit `main`.*
