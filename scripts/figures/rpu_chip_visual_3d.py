#!/usr/bin/env python3
"""Generate a detailed 3D visualization of the 64-tile RPU accelerator.

The script renders the physical chip layout including:
1. 8×8 tile array with internal layer stacks (DMA, Φ-RFT, SIS/NoC)
2. Spine modules (SIS hash engine, Feistel-48, unified controller)
3. DMA ingress region along south edge
4. PLL islands at NW/SE corners
5. Cascade/H3 signal routing overlays
6. Power grid visualization

Matches the floorplan from PHYSICAL_DESIGN_SPEC.md (8.5×8.5 mm die).

Usage:
    python tools/rpu_chip_visual_3d.py --output figures/rpu_chip_3d.png
    python tools/rpu_chip_visual_3d.py --output figures/rpu_chip_3d.png --show
    python tools/rpu_chip_visual_3d.py --detailed --output figures/rpu_chip_detailed.png
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap

# ============================================================================
# Physical constants from PHYSICAL_DESIGN_SPEC.md (scaled to mm)
# ============================================================================
DIE_SIZE = 8.5          # mm
CORE_MARGIN = 0.4       # mm (guard ring / ESD)
TILE_DIM = 8            # 8×8 grid
TILE_SIZE = 0.75        # mm per tile
TILE_GAP = 0.0625       # mm between tiles
GRID_ORIGIN_X = 0.5     # mm
GRID_ORIGIN_Y = 1.2     # mm

# Spine geometry
SPINE_X = 7.2           # mm
SPINE_WIDTH = 0.8       # mm

# Layer heights (Z dimension, scaled for visibility)
BASE_THICKNESS = 0.08   # mm (substrate)
TILE_HEIGHT = 0.6       # mm (full tile stack)
SPINE_HEIGHT = 0.7      # mm

# Layer stack within each tile (relative fractions)
LAYER_STACK: Sequence[Tuple[str, float, float, str]] = (
    ("DMA ingress / buffering", 0.00, 0.20, "#4c7eff"),
    ("CORDIC + kernel ROM", 0.20, 0.35, "#7c5cff"),
    ("Φ-RFT MAC array", 0.35, 0.65, "#ff9f45"),
    ("SIS digest", 0.65, 0.82, "#3ecf8e"),
    ("NoC router interface", 0.82, 1.00, "#2dd4bf"),
)

# SRAM layers (shown as darker insets)
SRAM_COLOR = "#1a1a2e"
SRAM_ALPHA = 0.85

# Spine module colors
SPINE_COLORS = {
    "sis_hash": "#e11d48",
    "feistel": "#8b5cf6",
    "controller": "#0ea5e9",
    "dma_ingress": "#06b6d4",
    "pll": "#fbbf24",
}

# Signal path definitions (col, row, z_offset)
CASCADE_PATH = [
    (0, 0, 1.0),
    (1, 1, 1.05),
    (2, 2, 1.1),
    (3, 3, 1.15),
    (4, 4, 1.2),
    (5, 5, 1.22),
    (6, 6, 1.25),
    (7, 7, 1.3),
]

H3_PATHS = [
    # Fan-out path 1 (diagonal SW-NE)
    [(0, 7, 1.0), (1, 6, 1.05), (2, 5, 1.1), (3, 4, 1.12)],
    # Fan-out path 2 (diagonal NW-SE)
    [(0, 0, 1.0), (2, 1, 1.08), (4, 2, 1.12), (6, 3, 1.15)],
    # Fan-out path 3 (horizontal mid)
    [(0, 4, 1.0), (2, 4, 1.05), (4, 4, 1.08), (6, 4, 1.1)],
]

# Power grid (simplified)
POWER_GRID_PITCH = 0.8  # mm


def prism_faces(x0: float, y0: float, x1: float, y1: float, z0: float, z1: float) -> List[List[Tuple[float, float, float]]]:
    """Return faces for a rectangular prism given corner coordinates."""
    faces = [
        [(x0, y0, z0), (x1, y0, z0), (x1, y1, z0), (x0, y1, z0)],  # bottom
        [(x0, y0, z1), (x1, y0, z1), (x1, y1, z1), (x0, y1, z1)],  # top
        [(x0, y0, z0), (x1, y0, z0), (x1, y0, z1), (x0, y0, z1)],  # front
        [(x0, y1, z0), (x1, y1, z0), (x1, y1, z1), (x0, y1, z1)],  # back
        [(x0, y0, z0), (x0, y1, z0), (x0, y1, z1), (x0, y0, z1)],  # left
        [(x1, y0, z0), (x1, y1, z0), (x1, y1, z1), (x1, y0, z1)],  # right
    ]
    return faces


def tile_position(col: int, row: int) -> Tuple[float, float]:
    """Get bottom-left corner of a tile in mm."""
    pitch = TILE_SIZE + TILE_GAP
    x = GRID_ORIGIN_X + col * pitch
    y = GRID_ORIGIN_Y + row * pitch
    return x, y


def add_substrate(ax) -> None:
    """Draw the silicon substrate / die base."""
    faces = prism_faces(0, 0, DIE_SIZE, DIE_SIZE, 0, BASE_THICKNESS)
    substrate = Poly3DCollection(
        faces,
        facecolors="#10141f",
        edgecolors="#333",
        linewidths=0.3,
        alpha=0.5,
    )
    ax.add_collection3d(substrate)


def add_guard_ring(ax) -> None:
    """Draw the ESD/TSV guard ring around die perimeter."""
    margin = CORE_MARGIN
    inner = margin
    outer = DIE_SIZE - margin
    ring_height = BASE_THICKNESS + 0.02
    
    # Four rectangular strips
    strips = [
        (0, 0, inner, DIE_SIZE),           # left
        (outer, 0, DIE_SIZE, DIE_SIZE),    # right
        (inner, 0, outer, inner),          # bottom
        (inner, outer, outer, DIE_SIZE),   # top
    ]
    for x0, y0, x1, y1 in strips:
        faces = prism_faces(x0, y0, x1, y1, BASE_THICKNESS, ring_height)
        ring = Poly3DCollection(
            faces,
            facecolors="#4a4a5a",
            edgecolors="#666",
            linewidths=0.2,
            alpha=0.7,
        )
        ax.add_collection3d(ring)


def add_tiles(ax, detailed: bool = False) -> None:
    """Render the 8×8 tile array with layer stacks."""
    for row in range(TILE_DIM):
        for col in range(TILE_DIM):
            x0, y0 = tile_position(col, row)
            x1, y1 = x0 + TILE_SIZE, y0 + TILE_SIZE
            
            for label, rel_z0, rel_z1, color in LAYER_STACK:
                z0 = BASE_THICKNESS + rel_z0 * TILE_HEIGHT
                z1 = BASE_THICKNESS + rel_z1 * TILE_HEIGHT
                faces = prism_faces(x0, y0, x1, y1, z0, z1)
                prism = Poly3DCollection(
                    faces,
                    facecolors=color,
                    edgecolors="#222",
                    linewidths=0.15 if detailed else 0.25,
                    alpha=0.88,
                )
                ax.add_collection3d(prism)
            
            # Add SRAM insets (visible in detailed mode)
            if detailed:
                # Scratch SRAM (bottom-left of tile)
                sram_x0 = x0 + 0.02
                sram_y0 = y0 + 0.02
                sram_x1 = x0 + 0.18
                sram_y1 = y0 + 0.12
                sram_z0 = BASE_THICKNESS + 0.35 * TILE_HEIGHT
                sram_z1 = sram_z0 + 0.08
                faces = prism_faces(sram_x0, sram_y0, sram_x1, sram_y1, sram_z0, sram_z1)
                sram = Poly3DCollection(
                    faces,
                    facecolors=SRAM_COLOR,
                    edgecolors="#444",
                    linewidths=0.1,
                    alpha=SRAM_ALPHA,
                )
                ax.add_collection3d(sram)
                
                # Topo SRAM (top of tile)
                topo_x0 = x0 + 0.02
                topo_y0 = y0 + 0.55
                topo_x1 = x0 + 0.25
                topo_y1 = y0 + 0.70
                faces = prism_faces(topo_x0, topo_y0, topo_x1, topo_y1, sram_z0, sram_z1)
                topo = Poly3DCollection(
                    faces,
                    facecolors=SRAM_COLOR,
                    edgecolors="#444",
                    linewidths=0.1,
                    alpha=SRAM_ALPHA,
                )
                ax.add_collection3d(topo)


def add_spine(ax) -> None:
    """Add the spine modules: SIS hash, Feistel, controller."""
    # SIS Hash Engine
    sis_y0, sis_y1 = 3.5, 5.5
    faces = prism_faces(SPINE_X, sis_y0, SPINE_X + SPINE_WIDTH, sis_y1,
                        BASE_THICKNESS, BASE_THICKNESS + SPINE_HEIGHT)
    sis = Poly3DCollection(
        faces,
        facecolors=SPINE_COLORS["sis_hash"],
        edgecolors="#500",
        linewidths=0.3,
        alpha=0.85,
    )
    ax.add_collection3d(sis)
    
    # Feistel-48
    feistel_y0, feistel_y1 = 2.0, 3.5
    faces = prism_faces(SPINE_X, feistel_y0, SPINE_X + SPINE_WIDTH, feistel_y1,
                        BASE_THICKNESS, BASE_THICKNESS + SPINE_HEIGHT * 0.8)
    feistel = Poly3DCollection(
        faces,
        facecolors=SPINE_COLORS["feistel"],
        edgecolors="#406",
        linewidths=0.3,
        alpha=0.85,
    )
    ax.add_collection3d(feistel)
    
    # Unified Controller
    ctrl_y0, ctrl_y1 = 5.5, 6.5
    faces = prism_faces(SPINE_X, ctrl_y0, SPINE_X + SPINE_WIDTH, ctrl_y1,
                        BASE_THICKNESS, BASE_THICKNESS + SPINE_HEIGHT * 0.5)
    ctrl = Poly3DCollection(
        faces,
        facecolors=SPINE_COLORS["controller"],
        edgecolors="#036",
        linewidths=0.3,
        alpha=0.85,
    )
    ax.add_collection3d(ctrl)


def add_dma_ingress(ax) -> None:
    """Add DMA ingress region along south edge."""
    dma_y1 = GRID_ORIGIN_Y - 0.1
    dma_y0 = CORE_MARGIN
    faces = prism_faces(CORE_MARGIN, dma_y0, DIE_SIZE - CORE_MARGIN, dma_y1,
                        BASE_THICKNESS, BASE_THICKNESS + 0.15)
    dma = Poly3DCollection(
        faces,
        facecolors=SPINE_COLORS["dma_ingress"],
        edgecolors="#055",
        linewidths=0.3,
        alpha=0.75,
    )
    ax.add_collection3d(dma)


def add_plls(ax) -> None:
    """Add PLL islands at NW and SE corners."""
    pll_size = 0.2
    pll_height = 0.12
    
    # PLL_NW
    pll_nw_x = CORE_MARGIN + 0.02
    pll_nw_y = DIE_SIZE - CORE_MARGIN - pll_size - 0.02
    faces = prism_faces(pll_nw_x, pll_nw_y, pll_nw_x + pll_size, pll_nw_y + pll_size,
                        BASE_THICKNESS, BASE_THICKNESS + pll_height)
    pll = Poly3DCollection(
        faces,
        facecolors=SPINE_COLORS["pll"],
        edgecolors="#960",
        linewidths=0.3,
        alpha=0.9,
    )
    ax.add_collection3d(pll)
    
    # PLL_SE
    pll_se_x = DIE_SIZE - CORE_MARGIN - pll_size - 0.02
    pll_se_y = CORE_MARGIN + 0.02
    faces = prism_faces(pll_se_x, pll_se_y, pll_se_x + pll_size, pll_se_y + pll_size,
                        BASE_THICKNESS, BASE_THICKNESS + pll_height)
    pll2 = Poly3DCollection(
        faces,
        facecolors=SPINE_COLORS["pll"],
        edgecolors="#960",
        linewidths=0.3,
        alpha=0.9,
    )
    ax.add_collection3d(pll2)


def add_signal_path(ax, points: Iterable[Tuple[int, int, float]], color: str, 
                    linewidth: float = 2.0, linestyle: str = '-') -> None:
    """Draw a polyline showing cascade/H3 routes over tiles."""
    coords = []
    for col, row, z_mult in points:
        x, y = tile_position(col, row)
        # Center of tile
        cx = x + TILE_SIZE / 2
        cy = y + TILE_SIZE / 2
        cz = BASE_THICKNESS + TILE_HEIGHT * z_mult
        coords.append((cx, cy, cz))
    
    xs = [c[0] for c in coords]
    ys = [c[1] for c in coords]
    zs = [c[2] for c in coords]
    ax.plot(xs, ys, zs, color=color, linewidth=linewidth, linestyle=linestyle, alpha=0.9)


def add_power_grid(ax) -> None:
    """Add simplified power grid visualization (horizontal stripes)."""
    z = BASE_THICKNESS + TILE_HEIGHT + 0.02
    for y in np.arange(CORE_MARGIN, DIE_SIZE - CORE_MARGIN, POWER_GRID_PITCH):
        ax.plot([CORE_MARGIN, DIE_SIZE - CORE_MARGIN], [y, y], [z, z],
                color='#666', linewidth=0.5, alpha=0.4)
    for x in np.arange(CORE_MARGIN, DIE_SIZE - CORE_MARGIN, POWER_GRID_PITCH):
        ax.plot([x, x], [CORE_MARGIN, DIE_SIZE - CORE_MARGIN], [z, z],
                color='#666', linewidth=0.5, alpha=0.4)


def build_legend(ax, detailed: bool = False) -> None:
    """Attach a 2D legend describing layers and modules."""
    patches = [mpatches.Patch(color=color, label=label) for label, _, _, color in LAYER_STACK]
    patches.append(mpatches.Patch(color=SPINE_COLORS["sis_hash"], label="SIS hash engine"))
    patches.append(mpatches.Patch(color=SPINE_COLORS["feistel"], label="Feistel-48 cipher"))
    patches.append(mpatches.Patch(color=SPINE_COLORS["controller"], label="Unified controller"))
    patches.append(mpatches.Patch(color=SPINE_COLORS["dma_ingress"], label="DMA ingress"))
    patches.append(mpatches.Patch(color=SPINE_COLORS["pll"], label="PLL islands"))
    patches.append(mpatches.Patch(color="#f54291", label="Cascade path"))
    patches.append(mpatches.Patch(color="#20c997", label="H3 fan-out"))
    if detailed:
        patches.append(mpatches.Patch(color=SRAM_COLOR, label="SRAM macros"))
    ax.legend(handles=patches, loc="upper left", bbox_to_anchor=(1.02, 0.98), fontsize=7)


def format_axes(ax, detailed: bool = False) -> None:
    """Configure axis appearance and viewpoint."""
    ax.set_box_aspect((1, 1, 0.35))
    
    # Tick marks at mm intervals
    ax.set_xticks(np.arange(0, DIE_SIZE + 1, 1))
    ax.set_yticks(np.arange(0, DIE_SIZE + 1, 1))
    ax.set_zticks([])
    
    ax.set_xlim(0, DIE_SIZE)
    ax.set_ylim(0, DIE_SIZE)
    ax.set_zlim(0, TILE_HEIGHT + 0.3)
    
    ax.set_xlabel("X (mm)", fontsize=8)
    ax.set_ylabel("Y (mm)", fontsize=8)
    ax.set_zlabel("")
    
    ax.view_init(elev=28, azim=-52)
    
    title = "QuantoniumOS RPU: 64-tile Φ-RFT Accelerator"
    subtitle = f"TSMC N7FF • {DIE_SIZE}×{DIE_SIZE} mm die • 950 MHz"
    if detailed:
        subtitle += " (detailed view)"
    ax.set_title(f"{title}\n{subtitle}", fontsize=10, pad=10)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a 3D visualization of the RPU chip layout."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("figures/rpu_chip_3d.png"),
        help="Path to write the rendered figure (default: figures/rpu_chip_3d.png)",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display the plot interactively after saving.",
    )
    parser.add_argument(
        "--detailed",
        action="store_true",
        help="Enable detailed view with SRAM macros and power grid.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=320,
        help="Output resolution (default: 320 DPI).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection="3d")
    ax.set_facecolor("#0d1117")
    fig.patch.set_facecolor("#0d1117")
    
    # Build the chip
    add_substrate(ax)
    add_guard_ring(ax)
    add_dma_ingress(ax)
    add_plls(ax)
    add_tiles(ax, detailed=args.detailed)
    add_spine(ax)
    
    # Signal overlays
    add_signal_path(ax, CASCADE_PATH, "#f54291", linewidth=2.5)
    for h3_path in H3_PATHS:
        add_signal_path(ax, h3_path, "#20c997", linewidth=1.8, linestyle="--")
    
    # Power grid (detailed mode)
    if args.detailed:
        add_power_grid(ax)
    
    # Finish up
    build_legend(ax, detailed=args.detailed)
    format_axes(ax, detailed=args.detailed)
    
    # Adjust tick colors for dark background
    ax.tick_params(colors='#888', labelsize=7)
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.xaxis.pane.set_edgecolor('#333')
    ax.yaxis.pane.set_edgecolor('#333')
    ax.zaxis.pane.set_edgecolor('#333')
    
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=args.dpi, bbox_inches="tight", 
                facecolor=fig.get_facecolor(), edgecolor='none')
    print(f"Saved: {args.output}")
    
    if args.show:
        plt.show()
    else:
        plt.close(fig)


if __name__ == "__main__":
    main()
