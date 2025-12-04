# Getting Started with RFTPU Physical Design

This guide will walk you through generating a **real chip layout** of the RFTPU.

## What You'll Get

After following this guide:
- ✅ GDS-II file (actual mask layout)
- ✅ Visual chip layout you can zoom into
- ✅ See individual tiles, NoC wiring, metal layers
- ✅ Timing and power reports
- ✅ Manufacturable design (with SkyWater SKY130)

## Prerequisites

### Option 1: Use Docker (Recommended)
```bash
# Pull OpenLane container
docker pull efabless/openlane:latest

# That's it! Container has everything.
```

### Option 2: Install Locally
```bash
# Clone OpenLane (can take time)
git clone --depth 1 https://github.com/The-OpenROAD-Project/OpenLane.git
cd OpenLane
make

# Install layout viewer
sudo apt-get install klayout
```

## Step-by-Step Instructions

### Step 1: Generate 4×4 RTL (30 seconds)

```bash
cd /workspaces/quantoniumos/hardware
python3 scripts/generate_4x4_variant.py
```

Expected output:
```
✓ Generated 4x4 variant: openlane/rftpu_4x4/src/rftpu_4x4_top.sv
  - Tiles: 8×8 → 4×4 (64 → 16)
  - Ready for OpenLane flow!
```

### Step 2: Choose Your Approach

#### Option A: Quick Preview with Yosys (5 minutes) - **Recommended First**
```bash
cd /workspaces/quantoniumos/hardware
python3 scripts/quick_synthesis.py
```
This gives you:
- Gate count estimate
- Cell statistics
- Synthesized netlist
- No full P&R (faster to see if design works)

#### Option B: Interactive Setup Script (Easy)
```bash
cd /workspaces/quantoniumos/hardware
./scripts/run_openlane.sh
```
This script will:
- Guide you through setup
- Handle Docker/PDK installation
- Provide multiple options
- Show you what's needed

#### Option C: Full OpenLane Docker (2-6 hours)
```bash
cd /workspaces/quantoniumos/hardware

# Run with PDK auto-install
docker run -it --rm \
  -v $(pwd):/workspace \
  efabless/openlane:latest bash -c "
    python3 -m volare fetch sky130 && \
    mkdir -p /openlane/designs/rftpu_4x4 && \
    cp -r /workspace/openlane/rftpu_4x4/* /openlane/designs/rftpu_4x4/ && \
    cd /openlane && \
    flow.tcl -design rftpu_4x4 -tag run_1 && \
    cp -r designs/rftpu_4x4/runs /workspace/openlane/rftpu_4x4/
  "
```

#### Option D: Local OpenLane Install
```bash
# First, install OpenLane
git clone https://github.com/The-OpenROAD-Project/OpenLane.git
cd OpenLane && make

# Then run
cd /path/to/OpenLane
cp -r /workspaces/quantoniumos/hardware/openlane/rftpu_4x4 designs/
./flow.tcl -design rftpu_4x4 -tag run_1
```

### Step 3: View the Layout (instant)

```bash
cd /workspaces/quantoniumos/hardware

# Quick launcher
./scripts/view_layout.sh

# Or directly with KLayout
klayout openlane/rftpu_4x4/runs/run_1/results/final/gds/rftpu_accelerator_4x4.gds
```

## What to Look For

Once KLayout opens:

### 1. Overview (zoom out)
- See the full 4×4 tile grid
- Square chip (~1.5mm × 1.5mm)
- Regular structure

### 2. Tile Level (zoom in on one tile)
- Each tile contains:
  - Φ-RFT core
  - NoC router
  - Control logic
  - Scratchpad memory

### 3. Interconnect (medium zoom)
- **Blue (M1)**: Local connections within tiles
- **Green (M2)**: Horizontal NoC channels
- **Red (M3)**: Vertical NoC channels
- **Purple (M4-M5)**: Power and clock

### 4. Cell Level (zoom way in)
- Individual standard cells
- NANDs, FFs, buffers
- Polysilicon gates

## Understanding the Reports

Check these files:

```bash
cd openlane/rftpu_4x4/runs/run_1/reports

# Area
cat synthesis/1-synthesis.stat.rpt.strategy4
# Look for: "Chip area for module"

# Timing
cat routing/22-parasitics_sta.rpt
# Look for: "slack (MET)" or "slack (VIOLATED)"

# Power (estimate)
cat routing/22-rcx_sta.power.rpt
# Look for: "Total Dynamic Power"
```

## Expected Results

### Successful Run Indicators
- ✅ `Flow complete` message
- ✅ GDS file exists
- ✅ Zero DRC violations (or very few)
- ✅ Timing slack positive
- ✅ No routing overflow

### Typical Metrics
```
Chip Area:        ~2-3 mm²
Gate Count:       ~150,000 gates
Clock Frequency:  100 MHz (10ns period)
Utilization:      30-40%
Runtime:          2-6 hours
Routing Layers:   M1-M5
```

## Troubleshooting

### "Synthesis failed"
```bash
# Check log
cat runs/run_1/logs/synthesis/1-synthesis.log

# Common fix: Check RTL syntax
verilator --lint-only openlane/rftpu_4x4/src/*.sv
```

### "Timing violations"
```bash
# Reduce clock frequency in config.json
"CLOCK_PERIOD": 15.0    # Was 10.0 → now 66 MHz instead of 100 MHz

# Or enable more optimization
"SYNTH_STRATEGY": "DELAY 0"
```

### "DRC violations"
```bash
# Check what rules failed
cat runs/run_1/reports/magic/magic.drc

# Often fixed by reducing density
"FP_CORE_UTIL": 25     # Was 30
"PL_TARGET_DENSITY": 0.35  # Was 0.40
```

### "Routing overflow"
```bash
# Allow more routing iterations
"GRT_OVERFLOW_ITERS": 200   # Was 150

# Or reduce congestion
"GRT_ADJUSTMENT": 0.4       # Was 0.3
```

## Next Steps

### 1. Analyze Your Layout
```bash
# Take screenshots
# Measure tile sizes
# Identify critical paths
```

### 2. Extract Metrics
```bash
# Parse reports
python3 scripts/extract_metrics.py runs/run_1
```

### 3. Optimize Design
```bash
# Adjust config
# Re-run flow
# Compare results
```

### 4. Scale to 8×8 (optional)
```bash
# Use full RTL
# Expect 12-24 hour runtime
# Need more powerful machine
```

## Tips for Success

1. **Start small**: 4×4 first, then scale up
2. **Monitor progress**: Check logs during run
3. **Save configs**: Keep working configurations
4. **Compare runs**: Use different tags (run_1, run_2, etc.)
5. **Document results**: Screenshots and metrics

## Help & Resources

- **OpenLane Docs**: https://openlane.readthedocs.io/
- **KLayout Manual**: https://klayout.de/doc/manual/
- **SkyWater PDK**: https://skywater-pdk.readthedocs.io/
- **Questions**: Open an issue on GitHub

---

**Ready to see your chip?** Start with Step 1! 🚀
