# Quick Start: See Your RFTPU Without Full P&R

Don't want to wait 2-6 hours? Here's how to get results **in 5 minutes**:

## The Fast Path ⚡

```bash
cd /workspaces/quantoniumos/hardware

# 1. Generate RTL (30 seconds)
python3 scripts/generate_4x4_variant.py

# 2. Quick synthesis (5 minutes)
python3 scripts/quick_synthesis.py

# Done! You'll see gate counts and cell statistics.
```

## What You Get

The quick synthesis gives you:
- ✅ **Gate count** - How many logic gates
- ✅ **Cell count** - Number of standard cells
- ✅ **Synthesized netlist** - Gate-level Verilog
- ✅ **Quick validation** - Does the design synthesize?

**No full place & route**, but enough to:
- Validate your design
- Get size estimates
- See if there are synthesis errors
- Decide if full P&R is worth it

## Full Flow (When You're Ready)

Once you've validated with quick synthesis:

```bash
# Interactive setup helper
./scripts/run_openlane.sh
```

This script will:
- Show you multiple options
- Handle Docker setup
- Guide you through PDK installation
- Let you choose: Docker, local, or web interface

## Comparison

| Method | Time | What You Get | When to Use |
|--------|------|--------------|-------------|
| **Quick Synthesis** | 5 min | Gate counts, netlist | First check, quick validation |
| **Interactive Script** | Setup | Guided flow | When you want help |
| **Full OpenLane** | 2-6 hrs | Complete layout (GDS) | Final chip view |

## Why Quick Synthesis First?

1. **Fast feedback** - Know if design works in minutes
2. **Catch errors early** - Before spending hours on P&R
3. **Estimate size** - See if 4×4 is right size
4. **No dependencies** - Just needs Yosys (auto-installed)

## Example Output

```
📊 Synthesis Results
══════════════════════════════════════════════
  Cells               : 147,234
  Wires               : 198,456
  Public wires        : 2,341
  Memories            : 512
  Processes           : 0

✓ Synthesis complete!
  Netlist: build/synthesis/rftpu_4x4_synth.v
  Log:     build/synthesis/yosys.log
```

## Next Steps

After quick synthesis:
- ✅ Design synthesizes → Try full OpenLane
- ❌ Synthesis errors → Fix RTL first
- ⚠️ Too big → Consider smaller variant

---

**Start here first!** Then decide if you want the full chip layout.
