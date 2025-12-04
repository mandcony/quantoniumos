# RFTPU OpenLane Physical Design

This directory contains configurations for generating **real chip layouts** of the RFTPU using open-source EDA tools.

## What is This?

Instead of just behavioral simulation, this flow produces:
- **GDS-II files** - Actual mask layouts a fab would use
- **DEF files** - Detailed placement of every gate
- **Timing reports** - With real wire delays
- **Power estimates** - Dynamic and static power
- **DRC/LVS clean** - Manufacturable design

## Directory Structure

```
openlane/
├── rftpu_4x4/           # 4×4 tile variant (recommended)
│   ├── config.json      # OpenLane configuration
│   ├── src/             # RTL sources
│   └── runs/            # Generated layouts
├── rftpu_8x8/           # Full 8×8 (optional, very slow)
└── README.md            # This file
```

## Quick Start

### 1. Generate 4×4 RTL
```bash
cd /workspaces/quantoniumos/hardware
python3 scripts/generate_4x4_variant.py
```

### 2. Run OpenLane
```bash
# Using Docker (easiest)
docker run -it -v $(pwd):/openlane \
  -v $(pwd)/openlane/rftpu_4x4:/openlane/designs/rftpu_4x4 \
  efabless/openlane:latest

# Inside container
flow.tcl -design rftpu_4x4 -tag run_1
```

### 3. View Layout
```bash
# Install KLayout if needed
sudo apt-get install klayout

# View the GDS
klayout openlane/rftpu_4x4/runs/run_1/results/final/gds/rftpu_accelerator_4x4.gds
```

## What You'll See

In KLayout or Magic, you'll see:
- **Tiles arranged in 4×4 grid**
- **NoC routing** between tiles (wormhole mesh)
- **Metal layers** (M1-M5)
  - M1: Local connections
  - M2: Horizontal routing
  - M3: Vertical routing  
  - M4-M5: Power/clock
- **Standard cells** (NAND, NOR, flip-flops, etc.)
- **Power grid** (VDD/VSS distribution)
- **Clock tree** branching to all tiles

## Design Metrics (4×4)

Expected results with SkyWater SKY130:
- **Core area**: ~2.25 mm²
- **Gate count**: ~150K gates
- **Tile count**: 16 tiles
- **Clock**: 100 MHz (10ns period)
- **Technology**: 130nm
- **Power**: ~50-100 mW (estimated)
- **Runtime**: 2-6 hours P&R

## Tools Used

1. **Yosys** - RTL synthesis (RTL → gates)
2. **OpenROAD** - Place & Route
   - Floorplanning
   - Placement
   - CTS (Clock Tree Synthesis)
   - Global routing
   - Detailed routing
3. **Magic** - DRC/LVS checks, GDS generation
4. **KLayout** - Layout viewing (recommended viewer)

## Why 4×4 Instead of 8×8?

| Aspect | 4×4 (16 tiles) | 8×8 (64 tiles) |
|--------|----------------|----------------|
| Runtime | 2-6 hours | 12-24 hours |
| Area | ~2.25 mm² | ~9 mm² |
| Congestion | Manageable | High |
| Routing | Converges easily | May need tuning |
| **Viewing** | **Clear, fast** | **Slow, zoomed out** |

The 4×4 variant is perfect for:
- ✅ Seeing the actual tile structure
- ✅ Understanding physical design
- ✅ Quick iteration
- ✅ Learning the tools

## Files Generated

After running OpenLane:

```
runs/run_1/
├── results/
│   ├── final/
│   │   ├── gds/
│   │   │   └── *.gds          ← VIEW THIS! (KLayout)
│   │   ├── def/
│   │   │   └── *.def          ← Placement data
│   │   └── verilog/gl/
│   │       └── *.v            ← Gate-level netlist
│   ├── synthesis/
│   │   └── *.v                ← Synthesized netlist
│   └── reports/
│       ├── synthesis/         ← Area, gate counts
│       ├── placement/         ← Utilization, density
│       ├── routing/           ← Congestion, DRCs
│       └── timing/            ← Setup/hold timing
└── logs/                      ← All tool logs
```

## Advanced: Tape-out Ready

To make this **actually manufacturable**:

1. **Add I/O pads** - Connect to chip pins
2. **Add seal ring** - Protects die edge
3. **LVS clean** - Layout vs. Schematic check
4. **DRC clean** - Design Rule Check
5. **Antenna rules** - Met
6. **ESD protection** - Added

Then submit to an open shuttle (e.g., Efabless) for actual fabrication!

## Resources

- [OpenLane Docs](https://openlane.readthedocs.io/)
- [SkyWater PDK](https://skywater-pdk.readthedocs.io/)
- [Efabless](https://efabless.com/) - Open MPW shuttles
- [FOSSi Foundation](https://www.fossi-foundation.org/) - Open silicon

## Next Steps

1. ✅ **Run the flow** - See what it produces
2. 📊 **Analyze results** - Check timing, area, power
3. 🔧 **Optimize** - Tweak config for better results
4. 🎨 **Visualize** - Take screenshots, make videos
5. 📦 **Document** - Share your silicon design!

---

*This is real chip design, not just simulation!* 🎉
