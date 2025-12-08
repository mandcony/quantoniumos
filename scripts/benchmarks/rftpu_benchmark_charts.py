#!/usr/bin/env python3
"""
RFTPU Benchmark Visualization
Generates charts from benchmark results
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from pathlib import Path

# Use a clean style
plt.style.use('seaborn-v0_8-whitegrid')

def create_power_scaling_chart():
    """Create DVFS power/efficiency chart."""
    
    modes = ['Ultra-Low', 'Low', 'Nominal', 'Boost', 'Turbo']
    freq_mhz = [285, 475, 665, 950, 1045]
    power_w = [1.2, 2.5, 4.5, 8.2, 10.2]
    gops = [715.9, 1193.2, 1670.5, 2386.4, 2625.0]
    efficiency = [582.0, 485.0, 370.4, 291.0, 256.1]
    
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    x = np.arange(len(modes))
    width = 0.35
    
    # Bar for GOPS
    bars1 = ax1.bar(x - width/2, gops, width, label='Throughput (GOPS)', 
                    color='#2ecc71', alpha=0.8)
    ax1.set_ylabel('Throughput (GOPS)', color='#2ecc71', fontsize=12)
    ax1.tick_params(axis='y', labelcolor='#2ecc71')
    ax1.set_ylim(0, 3000)
    
    # Secondary axis for efficiency
    ax2 = ax1.twinx()
    line = ax2.plot(x, efficiency, 'o-', color='#e74c3c', linewidth=2, 
                    markersize=10, label='Efficiency (GOPS/W)')
    ax2.set_ylabel('Efficiency (GOPS/W)', color='#e74c3c', fontsize=12)
    ax2.tick_params(axis='y', labelcolor='#e74c3c')
    ax2.set_ylim(0, 700)
    
    # Add power annotation on bars
    for i, (bar, pwr) in enumerate(zip(bars1, power_w)):
        ax1.annotate(f'{pwr}W', xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                     xytext=(0, 3), textcoords='offset points',
                     ha='center', va='bottom', fontsize=9)
    
    ax1.set_xlabel('Operating Mode', fontsize=12)
    ax1.set_xticks(x)
    ax1.set_xticklabels([f'{m}\n{f} MHz' for m, f in zip(modes, freq_mhz)])
    
    # Legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
    
    plt.title('RFTPU DVFS Power/Performance Scaling', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    return fig


def create_cascade_scaling_chart():
    """Create multi-chip cascade scaling chart."""
    
    chips = [1, 2, 4, 8, 16]
    tops = [2.39, 4.53, 8.78, 16.80, 31.31]
    efficiency = [277.2, 263.3, 255.0, 243.9, 227.3]
    overhead = [0.0, 5.0, 8.0, 12.0, 18.0]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Left: TOPS scaling
    bars = ax1.bar(range(len(chips)), tops, color=['#3498db', '#2980b9', '#1f618d', '#1a5276', '#154360'])
    ax1.set_xlabel('Number of Chips', fontsize=12)
    ax1.set_ylabel('Aggregate Throughput (TOPS)', fontsize=12)
    ax1.set_xticks(range(len(chips)))
    ax1.set_xticklabels(chips)
    ax1.set_title('Multi-Chip Cascade Throughput', fontsize=13, fontweight='bold')
    
    # Add value labels
    for bar, val in zip(bars, tops):
        ax1.annotate(f'{val:.1f}', xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                     xytext=(0, 3), textcoords='offset points',
                     ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    # Ideal scaling line
    ideal = [tops[0] * c for c in chips]
    ax1.plot(range(len(chips)), ideal, '--', color='gray', linewidth=1.5, 
             label='Ideal Linear Scaling')
    ax1.legend()
    
    # Right: Efficiency vs Overhead
    ax2.bar(range(len(chips)), efficiency, color='#27ae60', alpha=0.7, label='Efficiency (GOPS/W)')
    ax2_twin = ax2.twinx()
    ax2_twin.plot(range(len(chips)), overhead, 'o-', color='#c0392b', linewidth=2, 
                  markersize=8, label='Cascade Overhead (%)')
    
    ax2.set_xlabel('Number of Chips', fontsize=12)
    ax2.set_ylabel('Efficiency (GOPS/W)', color='#27ae60', fontsize=12)
    ax2_twin.set_ylabel('Cascade Overhead (%)', color='#c0392b', fontsize=12)
    ax2.set_xticks(range(len(chips)))
    ax2.set_xticklabels(chips)
    ax2.set_title('Cascade Efficiency & Overhead', fontsize=13, fontweight='bold')
    ax2.tick_params(axis='y', labelcolor='#27ae60')
    ax2_twin.tick_params(axis='y', labelcolor='#c0392b')
    ax2_twin.set_ylim(0, 25)
    
    # Combined legend
    lines1, labels1 = ax2.get_legend_handles_labels()
    lines2, labels2 = ax2_twin.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labels1 + labels2, loc='center right')
    
    plt.tight_layout()
    return fig


def create_fpga_comparison_chart():
    """Create FPGA comparison chart."""
    
    targets = ['RFTPU\nASIC (N7)', 'Xilinx\nVU13P', 'Xilinx\nVP1902', 
               'Intel\nAgilex F', 'Intel\nAgilex M']
    gops = [2386.4, 439.6, 942.0, 628.0, 1208.9]
    efficiency = [291.0, 5.9, 9.4, 7.4, 10.1]
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    colors = ['#e74c3c', '#3498db', '#9b59b6', '#1abc9c', '#f39c12']
    
    # Throughput comparison
    axes[0].barh(targets, gops, color=colors)
    axes[0].set_xlabel('Throughput (GOPS)', fontsize=12)
    axes[0].set_title('Raw Throughput', fontsize=13, fontweight='bold')
    axes[0].axvline(x=gops[0], color='#e74c3c', linestyle='--', alpha=0.5)
    for i, v in enumerate(gops):
        axes[0].text(v + 50, i, f'{v:.0f}', va='center', fontsize=10)
    
    # Efficiency comparison
    axes[1].barh(targets, efficiency, color=colors)
    axes[1].set_xlabel('Efficiency (GOPS/W)', fontsize=12)
    axes[1].set_title('Power Efficiency', fontsize=13, fontweight='bold')
    axes[1].axvline(x=efficiency[0], color='#e74c3c', linestyle='--', alpha=0.5)
    for i, v in enumerate(efficiency):
        axes[1].text(v + 5, i, f'{v:.1f}', va='center', fontsize=10)
    
    plt.suptitle('RFTPU ASIC vs FPGA Comparison', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    return fig


def create_workload_analysis_chart():
    """Create workload feasibility chart."""
    
    workloads = [
        'Audio 48kHz\nStereo',
        'Audio 192kHz\n5.1 Surround',
        'Pulse-Doppler\nRadar',
        'RF Spectrum\n2 GSPS',
        '5G NR OFDM\nProcessing',
        'SIS Hash\nCrypto'
    ]
    
    samples_rate_log = [np.log10(96e3), np.log10(1.152e6), np.log10(100e6), 
                        np.log10(2e9), np.log10(500e6), np.log10(1e9)]
    utilization = [0.0, 0.0, 0.2, 4.9, 1.2, 2.5]
    headroom = [100 - u for u in utilization]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Utilization bars
    x = np.arange(len(workloads))
    bars = ax1.bar(x, utilization, color='#3498db', alpha=0.8, label='Utilized')
    ax1.bar(x, headroom, bottom=utilization, color='#95a5a6', alpha=0.4, label='Headroom')
    
    ax1.set_xticks(x)
    ax1.set_xticklabels(workloads, fontsize=9)
    ax1.set_ylabel('Tile Utilization (%)', fontsize=12)
    ax1.set_title('RFTPU Utilization by Workload', fontsize=13, fontweight='bold')
    ax1.set_ylim(0, 110)
    ax1.axhline(y=100, color='red', linestyle='--', linewidth=1, alpha=0.7)
    ax1.legend(loc='upper right')
    
    # Add "OK" labels
    for i, bar in enumerate(bars):
        ax1.annotate('✓ OK', xy=(bar.get_x() + bar.get_width()/2, 105),
                     ha='center', va='bottom', fontsize=10, color='green', fontweight='bold')
    
    # Latency comparison
    latency_budget = [1000, 500, 10, 100, 50, 1]  # microseconds
    actual_latency = [0.013] * 6  # microseconds
    
    x_pos = np.arange(len(workloads))
    width = 0.35
    
    ax2.bar(x_pos - width/2, latency_budget, width, label='Latency Budget', 
            color='#e74c3c', alpha=0.7)
    ax2.bar(x_pos + width/2, actual_latency, width, label='Actual Latency', 
            color='#27ae60', alpha=0.8)
    
    ax2.set_yscale('log')
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(workloads, fontsize=9)
    ax2.set_ylabel('Latency (µs) - Log Scale', fontsize=12)
    ax2.set_title('Latency: Budget vs Actual', fontsize=13, fontweight='bold')
    ax2.legend(loc='upper right')
    
    plt.tight_layout()
    return fig


def create_radar_chart():
    """Create radar/spider chart comparing RFTPU to alternatives."""
    
    categories = ['Throughput', 'Efficiency', 'Latency', 'Cost', 'Scalability']
    
    # Normalized scores (0-100)
    rftpu = [85, 95, 98, 90, 85]
    cpu = [30, 10, 60, 95, 40]
    gpu = [95, 40, 20, 70, 60]
    fpga = [45, 15, 75, 30, 70]
    
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(projection='polar'))
    
    # Number of categories
    N = len(categories)
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]  # Complete the loop
    
    # Add data
    for data, name, color in [(rftpu, 'RFTPU ASIC', '#e74c3c'),
                               (cpu, 'CPU', '#3498db'),
                               (gpu, 'GPU', '#27ae60'),
                               (fpga, 'FPGA', '#9b59b6')]:
        values = data + data[:1]
        ax.plot(angles, values, 'o-', linewidth=2, label=name, color=color)
        ax.fill(angles, values, alpha=0.15, color=color)
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=11)
    ax.set_ylim(0, 100)
    ax.set_title('RFTPU vs Alternatives\n(Normalized Scores)', fontsize=14, fontweight='bold', pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
    
    plt.tight_layout()
    return fig


def main():
    """Generate all benchmark charts."""
    
    output_dir = Path('/workspaces/quantoniumos/figures')
    output_dir.mkdir(exist_ok=True)
    
    print("Generating RFTPU Benchmark Charts...")
    print("-" * 50)
    
    # Power Scaling
    fig = create_power_scaling_chart()
    fig.savefig(output_dir / 'rftpu_power_scaling.png', dpi=150, bbox_inches='tight')
    print(f"✓ Saved: {output_dir / 'rftpu_power_scaling.png'}")
    plt.close(fig)
    
    # Cascade Scaling
    fig = create_cascade_scaling_chart()
    fig.savefig(output_dir / 'rftpu_cascade_scaling.png', dpi=150, bbox_inches='tight')
    print(f"✓ Saved: {output_dir / 'rftpu_cascade_scaling.png'}")
    plt.close(fig)
    
    # FPGA Comparison
    fig = create_fpga_comparison_chart()
    fig.savefig(output_dir / 'rftpu_fpga_comparison.png', dpi=150, bbox_inches='tight')
    print(f"✓ Saved: {output_dir / 'rftpu_fpga_comparison.png'}")
    plt.close(fig)
    
    # Workload Analysis
    fig = create_workload_analysis_chart()
    fig.savefig(output_dir / 'rftpu_workload_analysis.png', dpi=150, bbox_inches='tight')
    print(f"✓ Saved: {output_dir / 'rftpu_workload_analysis.png'}")
    plt.close(fig)
    
    # Radar Chart
    fig = create_radar_chart()
    fig.savefig(output_dir / 'rftpu_radar_comparison.png', dpi=150, bbox_inches='tight')
    print(f"✓ Saved: {output_dir / 'rftpu_radar_comparison.png'}")
    plt.close(fig)
    
    print("-" * 50)
    print(f"All charts saved to: {output_dir}")


if __name__ == "__main__":
    main()
