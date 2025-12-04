#!/usr/bin/env python3
# SPDX-License-Identifier: LicenseRef-QuantoniumOS-Claims-NC
# Copyright (C) 2025 Luis M. Minier / quantoniumos
"""
Quick synthesis with Yosys - no full P&R needed
Just get gate count and initial area estimate
"""

import subprocess
import sys
from pathlib import Path
import re

def run_yosys_synthesis():
    """Run Yosys synthesis for quick preview."""
    
    script_dir = Path(__file__).parent
    hardware_dir = script_dir.parent
    design_dir = hardware_dir / "openlane" / "rftpu_4x4"
    rtl_file = design_dir / "src" / "rftpu_4x4_top.sv"
    build_dir = hardware_dir / "build" / "synthesis"
    
    build_dir.mkdir(parents=True, exist_ok=True)
    
    if not rtl_file.exists():
        print(f"❌ RTL not found: {rtl_file}")
        print("Run: python3 scripts/generate_4x4_variant.py")
        return False
    
    print("🔧 Running Yosys synthesis...")
    print(f"   Input: {rtl_file}")
    print(f"   Output: {build_dir}/")
    print()
    
    # Yosys script
    yosys_script = f"""
# Read design
read_verilog -sv {rtl_file}

# Elaborate
hierarchy -check -top rftpu_accelerator

# High-level synthesis
proc; opt; fsm; opt; memory; opt

# Technology mapping (generic)
techmap; opt

# Statistics
stat

# Write output
write_verilog {build_dir}/rftpu_4x4_synth.v
write_json {build_dir}/rftpu_4x4_synth.json
"""
    
    script_file = build_dir / "synth.ys"
    with open(script_file, 'w') as f:
        f.write(yosys_script)
    
    # Run Yosys
    try:
        result = subprocess.run(
            ['yosys', '-s', str(script_file)],
            capture_output=True,
            text=True,
            cwd=hardware_dir
        )
        
        log_file = build_dir / "yosys.log"
        with open(log_file, 'w') as f:
            f.write(result.stdout)
            f.write(result.stderr)
        
        print(result.stdout)
        
        # Parse statistics
        stats = parse_stats(result.stdout)
        
        print()
        print("=" * 60)
        print("📊 Synthesis Results")
        print("=" * 60)
        
        if stats:
            for key, value in stats.items():
                print(f"  {key:20s}: {value}")
        
        print()
        print(f"✓ Synthesis complete!")
        print(f"  Netlist: {build_dir}/rftpu_4x4_synth.v")
        print(f"  Log:     {build_dir}/yosys.log")
        
        return True
        
    except FileNotFoundError:
        print("❌ Yosys not found!")
        print("Install: sudo apt-get install yosys")
        return False
    except Exception as e:
        print(f"❌ Synthesis failed: {e}")
        return False

def parse_stats(log_output):
    """Extract key statistics from Yosys log."""
    stats = {}
    
    # Number of cells
    cell_match = re.search(r'Number of cells:\s+(\d+)', log_output)
    if cell_match:
        stats['Cells'] = cell_match.group(1)
    
    # Wires
    wire_match = re.search(r'Number of wires:\s+(\d+)', log_output)
    if wire_match:
        stats['Wires'] = wire_match.group(1)
    
    # Public wires
    public_match = re.search(r'Number of public wires:\s+(\d+)', log_output)
    if public_match:
        stats['Public wires'] = public_match.group(1)
    
    # Memories
    mem_match = re.search(r'Number of memories:\s+(\d+)', log_output)
    if mem_match:
        stats['Memories'] = mem_match.group(1)
    
    # Processes
    proc_match = re.search(r'Number of processes:\s+(\d+)', log_output)
    if proc_match:
        stats['Processes'] = proc_match.group(1)
    
    return stats

if __name__ == '__main__':
    success = run_yosys_synthesis()
    sys.exit(0 if success else 1)
