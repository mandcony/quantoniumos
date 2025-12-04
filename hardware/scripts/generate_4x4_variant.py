#!/usr/bin/env python3
# SPDX-License-Identifier: LicenseRef-QuantoniumOS-Claims-NC
# Copyright (C) 2025 Luis M. Minier / quantoniumos
"""
Generate 4x4 RFTPU variant for physical design
Reduces tile count from 8x8 (64 tiles) to 4x4 (16 tiles)
for faster OpenLane P&R and realistic chip viewing.
"""

import re
from pathlib import Path

def generate_4x4_variant():
    """Generate a 4x4 tile variant of the RFTPU architecture."""
    
    base_dir = Path(__file__).parent.parent
    input_file = base_dir / "rftpu_architecture.tlv"
    output_dir = base_dir / "openlane" / "rftpu_4x4" / "src"
    output_file = output_dir / "rftpu_4x4_top.sv"
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Reading from: {input_file}")
    print(f"Writing to: {output_file}")
    
    if not input_file.exists():
        print(f"Error: {input_file} not found!")
        return False
    
    with open(input_file, 'r') as f:
        content = f.read()
    
    # Extract only the SystemVerilog sections (between \SV markers)
    # Stop at the first appearance of m5_makerchip_module or TLV section
    sv_sections = []
    in_sv = False
    current_section = []
    
    for line in content.split('\n'):
        # Stop if we hit TLV testbench code
        if 'm5_makerchip_module' in line or line.strip() == '\\TLV':
            if in_sv and current_section:
                sv_sections.append('\n'.join(current_section))
            break
            
        if line.strip() == '\\SV':
            if in_sv:
                # End of SV section
                if current_section:
                    sv_sections.append('\n'.join(current_section))
                current_section = []
                in_sv = False
            else:
                # Start of SV section
                in_sv = True
        elif in_sv and not line.strip().startswith('\\'):
            current_section.append(line)
    
    # Add any remaining section
    if in_sv and current_section:
        sv_sections.append('\n'.join(current_section))
    
    if not sv_sections:
        print("Error: No SystemVerilog sections found!")
        return False
    
    # Combine all SV sections
    sv_content = '\n\n'.join(sv_sections)
    
    # Modify for 4x4 configuration
    modifications = [
        # Change TILE_DIM from 8 to 4
        (r'localparam int TILE_DIM\s*=\s*8;', 
         'localparam int TILE_DIM            = 4;  // Reduced for physical design'),
        
        # Change TILE_COUNT calculation (will be 16 instead of 64)
        (r'localparam int TILE_COUNT\s*=\s*TILE_DIM \* TILE_DIM;',
         'localparam int TILE_COUNT          = TILE_DIM * TILE_DIM;  // 4x4 = 16 tiles'),
        
        # Reduce MAX_INFLIGHT
        (r'localparam int MAX_INFLIGHT\s*=\s*64;',
         'localparam int MAX_INFLIGHT        = 16;  // Reduced for 4x4'),
        
        # Reduce memory depths
        (r'localparam int SCRATCH_DEPTH\s*=\s*64;',
         'localparam int SCRATCH_DEPTH       = 32;  // Reduced for area'),
        (r'localparam int TOPO_MEM_DEPTH\s*=\s*64;',
         'localparam int TOPO_MEM_DEPTH      = 32;  // Reduced for area'),
        
        # Change DMA_TILE_BITS from 7 to 4 (for 16 tiles)
        (r'localparam int DMA_TILE_BITS\s*=\s*7;',
         'localparam int DMA_TILE_BITS       = 4;  // log2(16) = 4'),
    ]
    
    modified_content = sv_content
    for pattern, replacement in modifications:
        modified_content = re.sub(pattern, replacement, modified_content)
    
    # Add header
    header = """// SPDX-License-Identifier: LicenseRef-QuantoniumOS-Claims-NC
// Copyright (C) 2025 Luis M. Minier / quantoniumos
// 
// RFTPU 4x4 Variant for Physical Design
// ======================================
// Reduced from 8x8 (64 tiles) to 4x4 (16 tiles) for:
// - Faster OpenLane place & route
// - Realistic chip layout viewing
// - Physical design validation
//
// Target: SkyWater SKY130 PDK
// Area estimate: ~2.25 mm² @ 130nm

`timescale 1ns/1ps

"""
    
    final_content = header + modified_content
    
    # Write output
    with open(output_file, 'w') as f:
        f.write(final_content)
    
    print(f"✓ Generated 4x4 variant: {output_file}")
    print(f"  - Tiles: 8×8 → 4×4 (64 → 16)")
    print(f"  - MAX_INFLIGHT: 64 → 16")
    print(f"  - Memory depths reduced")
    print(f"  - Ready for OpenLane flow!")
    
    return True

if __name__ == '__main__':
    import sys
    success = generate_4x4_variant()
    sys.exit(0 if success else 1)
