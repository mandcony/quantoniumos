# RFTPU 4x4 Physical Design Flow

This directory contains the OpenLane configuration for generating a real chip layout of a **4×4 tile RFTPU** using the SkyWater SKY130 open-source PDK.

## Overview

- **Design**: 4×4 RFTPU accelerator (16 tiles instead of 64)
- **PDK**: SkyWater SKY130 (130nm)
- **Tool Flow**: OpenLane (Yosys + OpenROAD + Magic)
- **Target**: GDS-II layout suitable for fabrication

## Prerequisites

### 1. Install OpenLane

```bash
# Clone OpenLane
cd /tmp
git clone --depth 1 https://github.com/The-OpenROAD-Project/OpenLane.git
cd OpenLane
make

# Or use Docker
docker pull efabless/openlane:latest
```

### 2. Install Viewers

#### KLayout (recommended)
```bash
sudo apt-get update
sudo apt-get install klayout
```

#### Magic
```bash
sudo apt-get install magic
```

## Directory Structure

```
rftpu_4x4/
├── config.json              # OpenLane JSON config
├── config_4x4.tcl          # TCL configuration (alternative)
├── src/                     # RTL sources (4x4 variant)
│   └── rftpu_4x4_top.sv
├── runs/                    # Generated runs
└── README.md
```

## Usage

### Step 1: Generate 4x4 RTL Variant

Create a reduced RFTPU with 4×4 tiles:

```bash
cd /workspaces/quantoniumos/hardware
python3 scripts/generate_4x4_variant.py
```

### Step 2: Run OpenLane Flow

```bash
# Using Docker (recommended)
docker run -it -v $(pwd):/openlane \
  -v $(pwd)/hardware/openlane/rftpu_4x4:/openlane/designs/rftpu_4x4 \
  efabless/openlane:latest

# Inside container
flow.tcl -design rftpu_4x4 -tag run_1

# Or use make
cd /path/to/OpenLane
make mount
./flow.tcl -design /openlane/designs/rftpu_4x4
```

### Step 3: View the Layout

#### Option A: KLayout (Color, Fast)
```bash
klayout hardware/openlane/rftpu_4x4/runs/run_1/results/final/gds/rftpu_accelerator_4x4.gds
```

#### Option B: Magic (DRC checks)
```bash
magic -T $PDK_ROOT/sky130A/libs.tech/magic/sky130A.tech \
  hardware/openlane/rftpu_4x4/runs/run_1/results/final/gds/rftpu_accelerator_4x4.gds
```

## Expected Results

### Area Estimate
- **Core area**: ~1.5mm × 1.5mm = 2.25 mm²
- **16 tiles**: ~0.14 mm² per tile
- **Clock frequency**: 100 MHz (10ns period)
- **Technology**: 130nm SKY130

### Layout Features Visible
- ✅ 4×4 tile grid arrangement
- ✅ NoC wormhole routing between tiles
- ✅ DMA ingress controller
- ✅ Power distribution network (PDN)
- ✅ Clock tree distribution
- ✅ Metal layers (M1-M5)
- ✅ Standard cell placement

## Outputs Generated

```
runs/run_1/results/
├── final/
│   ├── gds/
│   │   └── rftpu_accelerator_4x4.gds    # Final layout (view this!)
│   ├── def/
│   │   └── rftpu_accelerator_4x4.def    # Design exchange format
│   ├── lef/
│   │   └── rftpu_accelerator_4x4.lef    # Library exchange format
│   └── verilog/
│       └── gl/
│           └── rftpu_accelerator_4x4.v  # Gate-level netlist
├── synthesis/
│   └── rftpu_accelerator_4x4.v          # Synthesized netlist
└── reports/
    ├── synthesis/
    ├── placement/
    ├── routing/
    └── timing/
```

## Viewing Tips

### In KLayout
1. **Load GDS**: File → Open → Select `.gds` file
2. **Layer visibility**: Click eye icons to show/hide layers
3. **Metal layers**:
   - M1 (blue): Local routing
   - M2 (green): Horizontal routing
   - M3 (red): Vertical routing
   - M4-M5: Power/clock
4. **Zoom**: Scroll wheel or Shift+Drag
5. **Measure**: Ruler tool (View → Ruler)

### In Magic
```tcl
# Inside Magic console
load rftpu_accelerator_4x4
box
select area
what
see no *
see M2
see M3
drc check
```

## Scaling to Full 8×8

To generate the full 8×8 design (warning: much longer runtime):

```bash
# Increase die area in config.json
"DIE_AREA": "0 0 3000 3000"
"FP_CORE_UTIL": 35

# Use full 8x8 RTL
"VERILOG_FILES": ["dir::../../build/rftpu_architecture.sv"]

# Expect 12-24 hours for full P&R
```

## Troubleshooting

### DRC Violations
```bash
# Check DRC report
cat runs/run_1/reports/magic/magic.drc

# Common fixes: adjust routing settings
set ::env(GRT_ADJUSTMENT) 0.4
```

### Timing Violations
```bash
# Check timing report
cat runs/run_1/reports/routing/opensta_max.rpt

# Reduce clock frequency
set ::env(CLOCK_PERIOD) "15.0"  # 66 MHz
```

### Congestion
```bash
# Reduce utilization
set ::env(FP_CORE_UTIL) 25
set ::env(PL_TARGET_DENSITY) 0.35
```

## Next Steps

1. **Analyze the layout** - Identify tile placement patterns
2. **Extract parasitics** - Get realistic timing with RC delays
3. **Power analysis** - Estimate dynamic/static power
4. **Tape-out prep** - Add I/O pads, seal ring
5. **Fabrication** - Submit to SkyWater shuttle (if serious!)

## References

- [OpenLane Documentation](https://openlane.readthedocs.io/)
- [SkyWater PDK](https://skywater-pdk.readthedocs.io/)
- [KLayout Manual](https://www.klayout.de/doc/manual/)
- [Efabless Platform](https://efabless.com/) - For actual chip shuttles
