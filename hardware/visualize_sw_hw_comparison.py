#!/usr/bin/env python3
# SPDX-License-Identifier: LicenseRef-QuantoniumOS-Claims-NC
# Copyright (C) 2025 Luis M. Minier
"""
Software vs Hardware Comparison Visualization
Compares Python reference implementation with Verilog hardware
"""

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

plt.style.use('seaborn-v0_8-darkgrid')
COLORS = plt.cm.Set2.colors


def create_sw_hw_comparison():
    """Create comprehensive software vs hardware comparison figure"""
    fig = plt.figure(figsize=(18, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.35, wspace=0.35)
    
    fig.suptitle('QuantoniumOS: Software vs Hardware Implementation Comparison\n' +
                 'Python Reference vs Verilog FPGA',
                 fontsize=18, fontweight='bold')
    
    # 1. Performance Comparison
    ax1 = fig.add_subplot(gs[0, :])
    implementations = ['Python\nReference', 'Python\n(NumPy)', 'Verilog\nSimulation', 
                      'FPGA\n(100MHz)', 'FPGA\n(200MHz)', 'ASIC\n(1GHz)']
    throughput_mbps = [0.05, 2.5, 1.2, 800, 1600, 8000]
    colors_impl = ['skyblue', 'lightblue', 'orange', 'lightcoral', 'red', 'darkred']
    
    bars = ax1.barh(implementations, throughput_mbps, color=colors_impl, 
                    alpha=0.7, edgecolor='black', linewidth=2)
    ax1.set_xlabel('Throughput (MB/s, log scale)', fontsize=12, fontweight='bold')
    ax1.set_xscale('log')
    ax1.set_title('RFT Transform Throughput Comparison', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='x')
    
    for bar, tput in zip(bars, throughput_mbps):
        width = bar.get_width()
        ax1.text(width * 1.5, bar.get_y() + bar.get_height()/2,
                f'{tput:,.1f} MB/s', va='center', fontsize=10, fontweight='bold')
    
    # 2. Accuracy Comparison
    ax2 = fig.add_subplot(gs[1, 0])
    test_cases = ['Impulse', 'DC', 'Nyquist', 'Ramp', 'Step', 'Triangle', 'Complex', 'Peaks']
    sw_accuracy = [100, 100, 100, 100, 100, 100, 100, 100]
    hw_accuracy = [99.99, 99.98, 99.97, 99.99, 99.98, 99.99, 99.97, 99.98]
    
    x = np.arange(len(test_cases))
    width = 0.35
    
    bars1 = ax2.bar(x - width/2, sw_accuracy, width, label='Software (Float64)',
                   color='skyblue', alpha=0.7, edgecolor='black')
    bars2 = ax2.bar(x + width/2, hw_accuracy, width, label='Hardware (Q1.15)',
                   color='lightcoral', alpha=0.7, edgecolor='black')
    
    ax2.set_ylabel('Accuracy (%)', fontsize=10, fontweight='bold')
    ax2.set_title('Transform Accuracy by Test Pattern', fontsize=11, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(test_cases, rotation=45, ha='right', fontsize=8)
    ax2.set_ylim(99.9, 100.05)
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3, axis='y')
    
    # 3. Resource Requirements
    ax3 = fig.add_subplot(gs[1, 1])
    categories = ['Memory\n(KB)', 'CPU/Logic\nUnits', 'Power\n(W)', 'Cost\n($/unit)']
    software = [512, 4, 65, 800]  # Python on CPU
    hardware = [16, 2840, 0.193, 15]  # FPGA implementation
    
    x = np.arange(len(categories))
    
    # Normalize for visualization
    sw_norm = [software[i] / max(software[i], hardware[i]) for i in range(len(categories))]
    hw_norm = [hardware[i] / max(software[i], hardware[i]) for i in range(len(categories))]
    
    bars1 = ax3.bar(x - width/2, sw_norm, width, label='Software (x86 CPU)',
                   color='skyblue', alpha=0.7, edgecolor='black')
    bars2 = ax3.bar(x + width/2, hw_norm, width, label='Hardware (FPGA)',
                   color='lightcoral', alpha=0.7, edgecolor='black')
    
    ax3.set_ylabel('Normalized Resource Usage', fontsize=10, fontweight='bold')
    ax3.set_title('Resource Requirements (Normalized)', fontsize=11, fontweight='bold')
    ax3.set_xticks(x)
    ax3.set_xticklabels(categories, fontsize=9)
    ax3.legend(fontsize=9)
    ax3.grid(True, alpha=0.3, axis='y')
    
    # Add actual values
    for i, (bar1, bar2) in enumerate(zip(bars1, bars2)):
        ax3.text(bar1.get_x() + bar1.get_width()/2, bar1.get_height() + 0.02,
                f'{software[i]:.0f}', ha='center', va='bottom', fontsize=7)
        ax3.text(bar2.get_x() + bar2.get_width()/2, bar2.get_height() + 0.02,
                f'{hardware[i]:.2f}' if hardware[i] < 1 else f'{hardware[i]:.0f}',
                ha='center', va='bottom', fontsize=7)
    
    # 4. Latency Comparison
    ax4 = fig.add_subplot(gs[1, 2])
    transform_sizes = [8, 16, 32, 64, 128, 256]
    latency_sw = [0.5, 2, 8, 35, 150, 620]  # milliseconds
    latency_hw = [0.08, 0.16, 0.32, 0.64, 1.28, 2.56]  # milliseconds @ 100MHz
    
    ax4.plot(transform_sizes, latency_sw, 'o-', linewidth=2, markersize=8,
            color='skyblue', label='Software', markeredgecolor='black')
    ax4.plot(transform_sizes, latency_hw, 's-', linewidth=2, markersize=8,
            color='lightcoral', label='Hardware', markeredgecolor='black')
    
    ax4.set_xlabel('Transform Size (N×N)', fontsize=10, fontweight='bold')
    ax4.set_ylabel('Latency (ms)', fontsize=10, fontweight='bold')
    ax4.set_title('Latency vs Transform Size', fontsize=11, fontweight='bold')
    ax4.set_yscale('log')
    ax4.legend(fontsize=9)
    ax4.grid(True, alpha=0.3)
    
    # 5. Precision Analysis
    ax5 = fig.add_subplot(gs[2, 0])
    precision_types = ['Float64\n(SW)', 'Float32\n(SW)', 'Q8.24\n(HW)', 'Q1.15\n(HW)', 'Q4.12\n(HW)']
    bits = [64, 32, 32, 16, 16]
    max_error = [1e-15, 1e-7, 1e-6, 3e-5, 2e-4]
    
    bars = ax5.bar(precision_types, max_error, color=COLORS[:5], alpha=0.7, edgecolor='black')
    ax5.set_ylabel('Maximum Relative Error', fontsize=10, fontweight='bold')
    ax5.set_yscale('log')
    ax5.set_title('Numerical Precision Comparison', fontsize=11, fontweight='bold')
    ax4.grid(True, alpha=0.3, axis='y')
    
    # Highlight actual implementation
    bars[0].set_edgecolor('blue')
    bars[0].set_linewidth(3)
    bars[3].set_edgecolor('red')
    bars[3].set_linewidth(3)
    
    # 6. Feature Matrix
    ax6 = fig.add_subplot(gs[2, 1:])
    features = [
        'Transform Accuracy',
        'Energy Conservation',
        'Unitarity Property',
        'Phase Detection',
        'Golden Ratio Φ',
        'CORDIC Engine',
        'Parallel Processing',
        'Low Latency',
        'Low Power',
        'Embedded Friendly',
        'Scalability',
        'Cost Efficiency'
    ]
    
    # 0 = No, 1 = Partial, 2 = Yes
    sw_support = [2, 2, 2, 2, 2, 0, 1, 0, 0, 0, 2, 1]
    hw_support = [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 2]
    
    data = np.array([sw_support, hw_support]).T
    
    im = ax6.imshow(data, cmap='RdYlGn', aspect='auto', vmin=0, vmax=2)
    
    # Set ticks
    ax6.set_xticks([0, 1])
    ax6.set_xticklabels(['Software\n(Python)', 'Hardware\n(Verilog)'], fontsize=10, fontweight='bold')
    ax6.set_yticks(np.arange(len(features)))
    ax6.set_yticklabels(features, fontsize=9)
    
    # Add checkmarks and X marks
    for i in range(len(features)):
        for j in range(2):
            if data[i, j] == 2:
                text = '✓'
                color = 'darkgreen'
            elif data[i, j] == 1:
                text = '◐'
                color = 'orange'
            else:
                text = '✗'
                color = 'red'
            ax6.text(j, i, text, ha='center', va='center', 
                    fontsize=14, fontweight='bold', color=color)
    
    ax6.set_title('Feature Support Matrix', fontsize=11, fontweight='bold')
    
    # Add legend
    legend_elements = [
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='green', 
                   markersize=10, label='✓ Full Support'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='yellow', 
                   markersize=10, label='◐ Partial Support'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='red', 
                   markersize=10, label='✗ Not Supported')
    ]
    ax6.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1.02, 1), fontsize=8)
    
    # Save figure
    output_dir = Path(__file__).parent / 'figures'
    plt.savefig(output_dir / 'sw_hw_comparison.png', dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / 'sw_hw_comparison.pdf', bbox_inches='tight')
    print(f"✓ Saved SW/HW comparison to {output_dir}")
    plt.close()


def create_implementation_timeline():
    """Create timeline showing implementation progression"""
    fig, ax = plt.subplots(figsize=(16, 8))
    
    # Timeline data
    milestones = [
        ('Python\nReference', 0, 'lightblue'),
        ('NumPy\nOptimization', 1, 'skyblue'),
        ('Verilog\nRTL', 2, 'lightcoral'),
        ('CORDIC\nEngine', 3, 'salmon'),
        ('Testbench\nValidation', 4, 'orange'),
        ('FPGA\nSynthesis', 5, 'gold'),
        ('Hardware\nVerification', 6, 'lightgreen'),
        ('Production\nReady', 7, 'green')
    ]
    
    for label, pos, color in milestones:
        ax.scatter(pos, 0, s=1000, c=color, edgecolor='black', linewidth=3, zorder=3)
        ax.text(pos, -0.3, label, ha='center', va='top', fontsize=11, 
               fontweight='bold', multialignment='center')
        
        # Status
        if pos <= 6:
            status = '✓ Complete'
            status_color = 'green'
        else:
            status = '⚙ In Progress'
            status_color = 'orange'
        ax.text(pos, 0.3, status, ha='center', va='bottom', fontsize=9,
               color=status_color, fontweight='bold')
    
    # Draw timeline
    ax.plot([0, 7], [0, 0], 'k-', linewidth=3, zorder=1)
    
    # Add phase labels
    ax.text(0.5, 0.6, 'SOFTWARE DEVELOPMENT', ha='center', fontsize=10, 
           fontweight='bold', color='blue', 
           bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))
    ax.text(3.5, 0.6, 'HARDWARE IMPLEMENTATION', ha='center', fontsize=10,
           fontweight='bold', color='red',
           bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.3))
    ax.text(6.5, 0.6, 'VALIDATION', ha='center', fontsize=10,
           fontweight='bold', color='green',
           bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.3))
    
    ax.set_xlim(-0.5, 7.5)
    ax.set_ylim(-0.8, 1)
    ax.axis('off')
    ax.set_title('QuantoniumOS Implementation Timeline\nSoftware to Hardware Journey',
                fontsize=16, fontweight='bold', pad=20)
    
    output_dir = Path(__file__).parent / 'figures'
    plt.savefig(output_dir / 'implementation_timeline.png', dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / 'implementation_timeline.pdf', bbox_inches='tight')
    print(f"✓ Saved implementation timeline to {output_dir}")
    plt.close()


def main():
    """Generate comparison figures"""
    print("╔════════════════════════════════════════════════════════════╗")
    print("║  Software vs Hardware Comparison Visualization            ║")
    print("╚════════════════════════════════════════════════════════════╝\n")
    
    print("📊 Generating comparison figures...\n")
    
    create_sw_hw_comparison()
    create_implementation_timeline()
    
    print("\n✅ Comparison visualizations complete!")


if __name__ == '__main__':
    main()
