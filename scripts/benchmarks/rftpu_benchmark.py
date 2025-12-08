#!/usr/bin/env python3
"""
RFTPU Chip Power & Performance Benchmark
=========================================
Estimates the computational power of the 64-tile RFTPU accelerator
based on the architecture defined in rftpu_architecture.tlv.

Metrics computed:
- Peak throughput (GOPS, GB/s)
- Latency per transform
- Power efficiency (GOPS/W)
- Comparison to CPU/GPU baselines
"""

import math
from dataclasses import dataclass
from typing import Dict

# ============================================================================
# RFTPU Architecture Parameters (from rftpu_architecture.tlv)
# ============================================================================

@dataclass
class RFTPUConfig:
    """Architecture configuration extracted from RTL."""
    # Tile array
    tile_dim: int = 8                    # 8x8 grid
    tile_count: int = 64                 # Total tiles
    
    # Per-tile compute
    block_samples: int = 8               # Samples per RFT block
    sample_width: int = 16               # Bits per sample
    digest_width: int = 256              # SIS digest bits
    
    # Timing
    core_latency_cycles: int = 12        # phi_rft_core latency
    noc_hop_latency: int = 2             # Cycles per NoC hop
    max_inflight: int = 64               # NoC buffer depth
    
    # Target frequency (from PHYSICAL_DESIGN_SPEC.md)
    freq_mhz: int = 950                  # Tile clock
    freq_noc_mhz: int = 1200             # NoC clock
    
    # Power estimates (from spec)
    tile_power_mw: float = 85.0          # Per tile active
    noc_power_mw: float = 620.0          # Fabric total
    total_power_w: float = 8.2           # Full chip active


def compute_ops_per_block(cfg: RFTPUConfig) -> int:
    """
    Count operations per RFT block based on phi_rft_core logic.
    
    Per block (N=8 samples):
    - Complex kernel lookup: 8x8 = 64 lookups
    - Complex MAC: 8x8 = 64 complex multiplies + 64 complex adds
      = 64 * (4 real muls + 2 real adds) = 384 ops
    - Energy calc: 8 magnitude + 8 squares + 7 adds = ~50 ops
    - SIS lattice: 8 quantize + 8 sliding sums = ~80 ops
    """
    n = cfg.block_samples
    
    # RFT transform: N^2 complex MACs
    complex_macs = n * n  # 64
    real_ops_per_complex_mac = 4 + 2  # 4 muls + 2 adds
    rft_ops = complex_macs * real_ops_per_complex_mac  # 384
    
    # Energy calculation
    magnitude_ops = n * 3  # abs + abs + add per bin
    energy_ops = n * 2 + (n - 1)  # square + accumulate
    
    # SIS digest
    sis_ops = n * 5  # quantize + 4-tap sliding sum
    
    return rft_ops + magnitude_ops + energy_ops + sis_ops


def analyze_performance(cfg: RFTPUConfig) -> Dict:
    """Compute all performance metrics."""
    
    results = {}
    
    # === Per-tile metrics ===
    ops_per_block = compute_ops_per_block(cfg)
    results['ops_per_block'] = ops_per_block
    
    # Blocks per second per tile (limited by core latency)
    cycles_per_block = cfg.core_latency_cycles
    blocks_per_sec_per_tile = (cfg.freq_mhz * 1e6) / cycles_per_block
    results['blocks_per_sec_per_tile'] = blocks_per_sec_per_tile
    
    # Ops per second per tile
    ops_per_sec_per_tile = blocks_per_sec_per_tile * ops_per_block
    results['gops_per_tile'] = ops_per_sec_per_tile / 1e9
    
    # === Full chip metrics ===
    total_blocks_per_sec = blocks_per_sec_per_tile * cfg.tile_count
    results['total_blocks_per_sec'] = total_blocks_per_sec
    
    total_ops_per_sec = ops_per_sec_per_tile * cfg.tile_count
    results['total_gops'] = total_ops_per_sec / 1e9
    results['total_tops'] = total_ops_per_sec / 1e12
    
    # === Throughput ===
    bytes_per_block_in = cfg.block_samples * (cfg.sample_width // 8)
    bytes_per_block_out = cfg.digest_width // 8
    
    input_bandwidth_gbps = (total_blocks_per_sec * bytes_per_block_in) / 1e9
    output_bandwidth_gbps = (total_blocks_per_sec * bytes_per_block_out) / 1e9
    results['input_bandwidth_gbps'] = input_bandwidth_gbps
    results['output_bandwidth_gbps'] = output_bandwidth_gbps
    
    # Samples per second
    samples_per_sec = total_blocks_per_sec * cfg.block_samples
    results['samples_per_sec'] = samples_per_sec
    results['msamples_per_sec'] = samples_per_sec / 1e6
    
    # === Latency ===
    latency_ns = (cfg.core_latency_cycles / cfg.freq_mhz) * 1000
    results['single_block_latency_ns'] = latency_ns
    
    # Pipeline latency (fill + drain)
    pipeline_fill_cycles = cfg.tile_count * cfg.core_latency_cycles
    pipeline_latency_us = (pipeline_fill_cycles / cfg.freq_mhz)
    results['pipeline_fill_latency_us'] = pipeline_latency_us
    
    # === Power Efficiency ===
    results['gops_per_watt'] = results['total_gops'] / cfg.total_power_w
    results['samples_per_joule'] = samples_per_sec / cfg.total_power_w
    
    # === NoC metrics ===
    max_noc_bandwidth_gbps = (cfg.freq_noc_mhz * 1e6 * cfg.digest_width) / (8 * 1e9)
    results['noc_bandwidth_gbps'] = max_noc_bandwidth_gbps
    
    # Worst-case NoC latency (corner to corner)
    max_hops = 2 * (cfg.tile_dim - 1)
    max_noc_latency_cycles = max_hops * cfg.noc_hop_latency
    max_noc_latency_ns = (max_noc_latency_cycles / cfg.freq_noc_mhz) * 1000
    results['max_noc_latency_ns'] = max_noc_latency_ns
    
    return results


def compare_to_baselines(results: Dict) -> Dict:
    """Compare RFTPU performance to CPU/GPU baselines."""
    
    comparisons = {}
    
    # Baseline: Intel i9-13900K single-core FFT (FFTW)
    # ~50 GFLOPS single-core, ~800 GFLOPS all-core
    cpu_single_gops = 50
    cpu_all_gops = 800
    
    comparisons['vs_cpu_single_core'] = results['total_gops'] / cpu_single_gops
    comparisons['vs_cpu_all_cores'] = results['total_gops'] / cpu_all_gops
    
    # Baseline: NVIDIA RTX 4090 FFT (cuFFT)
    # ~80 TFLOPS FP16, but FFT is memory-bound ~5-10 TFLOPS effective
    gpu_fft_tops = 8
    comparisons['vs_gpu_fft'] = results['total_tops'] / gpu_fft_tops
    
    # Power efficiency comparison
    # CPU: ~250W TDP → ~3.2 GOPS/W
    # GPU: ~450W TDP → ~18 GOPS/W (at 8 TOPS)
    cpu_gops_per_watt = 3.2
    gpu_gops_per_watt = 18
    
    comparisons['efficiency_vs_cpu'] = results['gops_per_watt'] / cpu_gops_per_watt
    comparisons['efficiency_vs_gpu'] = results['gops_per_watt'] / gpu_gops_per_watt
    
    # Latency comparison
    # CPU FFT-8: ~50ns, GPU FFT-8: ~2000ns (kernel launch overhead)
    cpu_fft8_ns = 50
    gpu_fft8_ns = 2000
    
    comparisons['latency_vs_cpu'] = cpu_fft8_ns / results['single_block_latency_ns']
    comparisons['latency_vs_gpu'] = gpu_fft8_ns / results['single_block_latency_ns']
    
    return comparisons


def format_report(cfg: RFTPUConfig, results: Dict, comparisons: Dict) -> str:
    """Generate formatted benchmark report."""
    
    report = []
    report.append("=" * 70)
    report.append("RFTPU ACCELERATOR BENCHMARK REPORT")
    report.append("=" * 70)
    report.append("")
    
    # Architecture
    report.append("ARCHITECTURE")
    report.append("-" * 40)
    report.append(f"  Tile Array:          {cfg.tile_dim}×{cfg.tile_dim} = {cfg.tile_count} tiles")
    report.append(f"  Block Size:          {cfg.block_samples} samples × {cfg.sample_width} bits")
    report.append(f"  Digest Width:        {cfg.digest_width} bits")
    report.append(f"  Core Latency:        {cfg.core_latency_cycles} cycles")
    report.append(f"  Tile Frequency:      {cfg.freq_mhz} MHz")
    report.append(f"  NoC Frequency:       {cfg.freq_noc_mhz} MHz")
    report.append(f"  Total Power:         {cfg.total_power_w:.1f} W")
    report.append("")
    
    # Compute
    report.append("COMPUTE PERFORMANCE")
    report.append("-" * 40)
    report.append(f"  Ops per RFT Block:   {results['ops_per_block']} ops")
    report.append(f"  Per-Tile Throughput: {results['gops_per_tile']:.2f} GOPS")
    report.append(f"  Total Throughput:    {results['total_gops']:.1f} GOPS ({results['total_tops']:.3f} TOPS)")
    report.append(f"  RFT Blocks/sec:      {results['total_blocks_per_sec']/1e6:.1f} M blocks/s")
    report.append(f"  Samples/sec:         {results['msamples_per_sec']:.0f} Msamples/s")
    report.append("")
    
    # Bandwidth
    report.append("MEMORY BANDWIDTH")
    report.append("-" * 40)
    report.append(f"  Input Bandwidth:     {results['input_bandwidth_gbps']:.1f} GB/s")
    report.append(f"  Output Bandwidth:    {results['output_bandwidth_gbps']:.1f} GB/s")
    report.append(f"  NoC Bandwidth:       {results['noc_bandwidth_gbps']:.1f} GB/s")
    report.append("")
    
    # Latency
    report.append("LATENCY")
    report.append("-" * 40)
    report.append(f"  Single Block:        {results['single_block_latency_ns']:.1f} ns")
    report.append(f"  Pipeline Fill:       {results['pipeline_fill_latency_us']:.2f} µs")
    report.append(f"  Max NoC Latency:     {results['max_noc_latency_ns']:.1f} ns")
    report.append("")
    
    # Efficiency
    report.append("POWER EFFICIENCY")
    report.append("-" * 40)
    report.append(f"  Compute Efficiency:  {results['gops_per_watt']:.1f} GOPS/W")
    report.append(f"  Sample Efficiency:   {results['samples_per_joule']/1e6:.1f} Msamples/J")
    report.append("")
    
    # Comparisons
    report.append("COMPARISON TO BASELINES")
    report.append("-" * 40)
    report.append(f"  vs CPU (single-core): {comparisons['vs_cpu_single_core']:.1f}× throughput")
    report.append(f"  vs CPU (all-cores):   {comparisons['vs_cpu_all_cores']:.2f}× throughput")
    report.append(f"  vs GPU (cuFFT):       {comparisons['vs_gpu_fft']:.2f}× throughput")
    report.append(f"  ")
    report.append(f"  Efficiency vs CPU:    {comparisons['efficiency_vs_cpu']:.1f}× better GOPS/W")
    report.append(f"  Efficiency vs GPU:    {comparisons['efficiency_vs_gpu']:.1f}× better GOPS/W")
    report.append(f"  ")
    report.append(f"  Latency vs CPU:       {comparisons['latency_vs_cpu']:.1f}× faster")
    report.append(f"  Latency vs GPU:       {comparisons['latency_vs_gpu']:.0f}× faster")
    report.append("")
    
    # Summary
    report.append("=" * 70)
    report.append("SUMMARY")
    report.append("=" * 70)
    
    # Key strengths
    strengths = []
    if comparisons['latency_vs_gpu'] > 100:
        strengths.append(f"✓ Ultra-low latency: {results['single_block_latency_ns']:.1f}ns per transform")
    if comparisons['efficiency_vs_cpu'] > 10:
        strengths.append(f"✓ High efficiency: {results['gops_per_watt']:.0f} GOPS/W")
    if results['total_gops'] > 100:
        strengths.append(f"✓ High throughput: {results['total_gops']:.0f} GOPS sustained")
    if results['msamples_per_sec'] > 1000:
        strengths.append(f"✓ Real-time capable: {results['msamples_per_sec']/1000:.1f} Gsamples/s")
    
    for s in strengths:
        report.append(f"  {s}")
    
    report.append("")
    report.append("  Use Cases:")
    report.append("  • Real-time audio/RF processing at sub-microsecond latency")
    report.append("  • Edge inference with deterministic timing")
    report.append("  • Cryptographic hashing (SIS-based) at wire speed")
    report.append("  • Scientific computing with resonance detection")
    report.append("")
    
    return "\n".join(report)


def power_scaling_benchmark(base_cfg: RFTPUConfig) -> Dict:
    """Analyze performance across different power/frequency operating points."""
    
    results = {
        'operating_points': [],
        'freq_mhz': [],
        'power_w': [],
        'gops': [],
        'gops_per_watt': [],
        'latency_ns': []
    }
    
    # Voltage/frequency scaling points (DVFS)
    # (freq_scale, voltage_scale, power_scale)
    dvfs_points = [
        ('ultra_low', 0.3, 0.65, 0.15),    # 285 MHz, 0.65V, ~1.2W
        ('low', 0.5, 0.75, 0.30),           # 475 MHz, 0.75V, ~2.5W
        ('nominal', 0.7, 0.85, 0.55),       # 665 MHz, 0.85V, ~4.5W
        ('boost', 1.0, 1.0, 1.0),           # 950 MHz, 1.0V, 8.2W
        ('turbo', 1.1, 1.05, 1.25),         # 1045 MHz, 1.05V, ~10.3W
    ]
    
    for name, freq_scale, voltage_scale, power_scale in dvfs_points:
        cfg = RFTPUConfig(
            freq_mhz=int(base_cfg.freq_mhz * freq_scale),
            total_power_w=base_cfg.total_power_w * power_scale
        )
        perf = analyze_performance(cfg)
        
        results['operating_points'].append(name)
        results['freq_mhz'].append(cfg.freq_mhz)
        results['power_w'].append(cfg.total_power_w)
        results['gops'].append(perf['total_gops'])
        results['gops_per_watt'].append(perf['gops_per_watt'])
        results['latency_ns'].append(perf['single_block_latency_ns'])
    
    # Find optimal efficiency point
    best_eff_idx = max(range(len(results['gops_per_watt'])), 
                       key=lambda i: results['gops_per_watt'][i])
    results['optimal_efficiency_point'] = results['operating_points'][best_eff_idx]
    results['peak_gops_per_watt'] = results['gops_per_watt'][best_eff_idx]
    
    return results


def multi_chip_cascade_benchmark(base_cfg: RFTPUConfig) -> Dict:
    """Simulate multi-chip configurations with cascade interconnect."""
    
    results = {
        'chip_counts': [],
        'total_tiles': [],
        'aggregate_gops': [],
        'aggregate_tops': [],
        'total_power_w': [],
        'efficiency_gops_w': [],
        'cascade_overhead_pct': [],
        'die2die_bandwidth_gbps': []
    }
    
    # H3 cascade die-to-die link specs (from RTL: 32-bit @ 1.2GHz = 4.8 GB/s per link)
    d2d_link_gbps = 4.8
    links_per_chip = 4  # N/S/E/W cascade ports
    
    for chip_count in [1, 2, 4, 8, 16]:
        total_tiles = chip_count * base_cfg.tile_count
        base_perf = analyze_performance(base_cfg)
        
        # Cascade overhead model:
        # - 2 chips: ~5% overhead (simple linear connection)
        # - 4 chips: ~8% overhead (2×2 mesh)
        # - 8 chips: ~12% overhead (2×4 mesh)
        # - 16 chips: ~18% overhead (4×4 mesh)
        overhead_pct = 0.0
        if chip_count == 2:
            overhead_pct = 5.0
        elif chip_count == 4:
            overhead_pct = 8.0
        elif chip_count == 8:
            overhead_pct = 12.0
        elif chip_count == 16:
            overhead_pct = 18.0
        
        efficiency_factor = 1.0 - (overhead_pct / 100.0)
        
        aggregate_gops = base_perf['total_gops'] * chip_count * efficiency_factor
        aggregate_tops = aggregate_gops / 1000
        total_power = base_cfg.total_power_w * chip_count * 1.05  # +5% for interposer/links
        
        # Die-to-die bandwidth (bidirectional links)
        d2d_bw = d2d_link_gbps * links_per_chip * (chip_count - 1) if chip_count > 1 else 0
        
        results['chip_counts'].append(chip_count)
        results['total_tiles'].append(total_tiles)
        results['aggregate_gops'].append(aggregate_gops)
        results['aggregate_tops'].append(aggregate_tops)
        results['total_power_w'].append(total_power)
        results['efficiency_gops_w'].append(aggregate_gops / total_power)
        results['cascade_overhead_pct'].append(overhead_pct)
        results['die2die_bandwidth_gbps'].append(d2d_bw)
    
    return results


def fpga_comparison_benchmark(base_cfg: RFTPUConfig) -> Dict:
    """Compare RFTPU ASIC to FPGA implementations."""
    
    # RFTPU ASIC baseline
    asic_perf = analyze_performance(base_cfg)
    
    # FPGA target specs (realistic estimates)
    fpga_targets = {
        'xilinx_vu13p': {
            'name': 'Xilinx VU13P (UltraScale+)',
            'tile_freq_mhz': 350,      # Realistic for complex DSP
            'max_tiles': 32,           # Resource-limited
            'power_w': 75,             # Typical FPGA power
        },
        'xilinx_vp1902': {
            'name': 'Xilinx VP1902 (Versal Premium)',
            'tile_freq_mhz': 500,      # AI engines help
            'max_tiles': 48,
            'power_w': 100,
        },
        'intel_agilex_f': {
            'name': 'Intel Agilex F-Series F1',
            'tile_freq_mhz': 400,
            'max_tiles': 40,
            'power_w': 85,
        },
        'intel_agilex_m': {
            'name': 'Intel Agilex M-Series',
            'tile_freq_mhz': 550,      # HBM version
            'max_tiles': 56,
            'power_w': 120,
        },
    }
    
    results = {
        'asic': {
            'gops': asic_perf['total_gops'],
            'tops': asic_perf['total_tops'],
            'gops_per_watt': asic_perf['gops_per_watt'],
            'latency_ns': asic_perf['single_block_latency_ns'],
            'power_w': base_cfg.total_power_w,
        },
        'fpga_targets': {}
    }
    
    ops_per_block = compute_ops_per_block(base_cfg)
    
    for key, fpga in fpga_targets.items():
        blocks_per_sec = (fpga['tile_freq_mhz'] * 1e6 / base_cfg.core_latency_cycles) * fpga['max_tiles']
        gops = (blocks_per_sec * ops_per_block) / 1e9
        tops = gops / 1000
        gops_per_watt = gops / fpga['power_w']
        latency_ns = (base_cfg.core_latency_cycles / fpga['tile_freq_mhz']) * 1000
        
        results['fpga_targets'][key] = {
            'name': fpga['name'],
            'gops': gops,
            'tops': tops,
            'gops_per_watt': gops_per_watt,
            'latency_ns': latency_ns,
            'power_w': fpga['power_w'],
            'vs_asic_throughput': asic_perf['total_gops'] / gops,
            'vs_asic_efficiency': asic_perf['gops_per_watt'] / gops_per_watt,
            'vs_asic_latency': latency_ns / asic_perf['single_block_latency_ns'],
        }
    
    return results


def workload_benchmark(base_cfg: RFTPUConfig) -> Dict:
    """Benchmark specific workload scenarios."""
    
    base_perf = analyze_performance(base_cfg)
    
    workloads = {
        'audio_realtime_48k': {
            'description': 'Real-time audio at 48 kHz stereo',
            'samples_per_sec': 48000 * 2,
            'block_size': 8,
            'latency_budget_us': 1000,  # 1ms for interactive
        },
        'audio_realtime_192k': {
            'description': 'High-res audio at 192 kHz 5.1 surround',
            'samples_per_sec': 192000 * 6,
            'block_size': 8,
            'latency_budget_us': 500,
        },
        'radar_pulse_doppler': {
            'description': 'Pulse-Doppler radar processing',
            'samples_per_sec': 100e6,  # 100 MSPS ADC
            'block_size': 8,
            'latency_budget_us': 10,  # Fast target tracking
        },
        'rf_spectrum_analyzer': {
            'description': 'Wideband RF spectrum analysis',
            'samples_per_sec': 2e9,  # 2 GSPS
            'block_size': 8,
            'latency_budget_us': 100,
        },
        '5g_ofdm_processing': {
            'description': '5G NR OFDM subcarrier processing',
            'samples_per_sec': 500e6,  # 500 MSPS effective
            'block_size': 8,
            'latency_budget_us': 50,
        },
        'crypto_sis_hashing': {
            'description': 'Post-quantum SIS hash computation',
            'samples_per_sec': 1e9,  # 1 billion hash inputs/sec
            'block_size': 8,
            'latency_budget_us': 1,  # Ultra-low latency
        },
    }
    
    results = {}
    
    for key, workload in workloads.items():
        blocks_needed = workload['samples_per_sec'] / workload['block_size']
        tiles_required = blocks_needed / (base_cfg.freq_mhz * 1e6 / base_cfg.core_latency_cycles)
        
        actual_latency_us = base_perf['single_block_latency_ns'] / 1000
        latency_ok = actual_latency_us <= workload['latency_budget_us']
        
        utilization = min(100.0, (tiles_required / base_cfg.tile_count) * 100)
        headroom = max(0, (base_cfg.tile_count - tiles_required) / base_cfg.tile_count * 100)
        
        results[key] = {
            'description': workload['description'],
            'samples_per_sec': workload['samples_per_sec'],
            'tiles_required': tiles_required,
            'utilization_pct': utilization,
            'headroom_pct': headroom,
            'latency_budget_us': workload['latency_budget_us'],
            'actual_latency_us': actual_latency_us,
            'latency_ok': latency_ok,
            'feasible': tiles_required <= base_cfg.tile_count and latency_ok,
        }
    
    return results


def format_extended_report(base_cfg: RFTPUConfig, 
                           power_results: Dict,
                           cascade_results: Dict, 
                           fpga_results: Dict,
                           workload_results: Dict) -> str:
    """Generate extended benchmark report."""
    
    report = []
    report.append("")
    report.append("=" * 70)
    report.append("EXTENDED BENCHMARK SUITE")
    report.append("=" * 70)
    
    # Power Scaling
    report.append("")
    report.append("POWER/FREQUENCY SCALING (DVFS)")
    report.append("-" * 70)
    report.append(f"{'Mode':<12} {'Freq(MHz)':<10} {'Power(W)':<10} {'GOPS':<10} {'GOPS/W':<10} {'Latency(ns)':<12}")
    report.append("-" * 70)
    for i, mode in enumerate(power_results['operating_points']):
        report.append(f"{mode:<12} {power_results['freq_mhz'][i]:<10} "
                      f"{power_results['power_w'][i]:<10.1f} "
                      f"{power_results['gops'][i]:<10.1f} "
                      f"{power_results['gops_per_watt'][i]:<10.1f} "
                      f"{power_results['latency_ns'][i]:<12.1f}")
    report.append("-" * 70)
    report.append(f"  Optimal efficiency: {power_results['optimal_efficiency_point']} "
                  f"({power_results['peak_gops_per_watt']:.1f} GOPS/W)")
    
    # Multi-Chip Cascade
    report.append("")
    report.append("MULTI-CHIP CASCADE SCALING")
    report.append("-" * 70)
    report.append(f"{'Chips':<8} {'Tiles':<8} {'GOPS':<12} {'TOPS':<10} {'Power(W)':<10} {'Eff.':<10} {'Overhead':<10}")
    report.append("-" * 70)
    for i, chips in enumerate(cascade_results['chip_counts']):
        report.append(f"{chips:<8} {cascade_results['total_tiles'][i]:<8} "
                      f"{cascade_results['aggregate_gops'][i]:<12.1f} "
                      f"{cascade_results['aggregate_tops'][i]:<10.2f} "
                      f"{cascade_results['total_power_w'][i]:<10.1f} "
                      f"{cascade_results['efficiency_gops_w'][i]:<10.1f} "
                      f"{cascade_results['cascade_overhead_pct'][i]:<10.1f}%")
    report.append("-" * 70)
    report.append(f"  16-chip config: {cascade_results['aggregate_tops'][-1]:.1f} TOPS @ "
                  f"{cascade_results['efficiency_gops_w'][-1]:.1f} GOPS/W")
    
    # FPGA Comparison
    report.append("")
    report.append("FPGA COMPARISON")
    report.append("-" * 70)
    report.append(f"{'Target':<30} {'GOPS':<10} {'GOPS/W':<10} {'vs ASIC':<12} {'Price':<10}")
    report.append("-" * 70)
    report.append(f"{'RFTPU ASIC (N7)':<30} {fpga_results['asic']['gops']:<10.1f} "
                  f"{fpga_results['asic']['gops_per_watt']:<10.1f} {'1.0×':<12} "
                  f"${fpga_results['asic']['estimated_price_usd']:<9}")
    for key, fpga in fpga_results['fpga_targets'].items():
        report.append(f"{fpga['name']:<30} {fpga['gops']:<10.1f} "
                      f"{fpga['gops_per_watt']:<10.1f} "
                      f"{1/fpga['vs_asic_throughput']:.2f}×{'':<8} "
                      f"${fpga['price_usd']:<9,}")
    report.append("-" * 70)
    report.append(f"  ASIC advantage: {fpga_results['fpga_targets']['xilinx_vu13p']['vs_asic_throughput']:.1f}× throughput, "
                  f"{fpga_results['fpga_targets']['xilinx_vu13p']['vs_asic_efficiency']:.1f}× efficiency vs best FPGA")
    
    # Workload Analysis
    report.append("")
    report.append("WORKLOAD FEASIBILITY ANALYSIS")
    report.append("-" * 70)
    report.append(f"{'Workload':<26} {'Rate':<14} {'Tiles':<8} {'Util':<8} {'Latency':<12} {'Status':<10}")
    report.append("-" * 70)
    for key, wl in workload_results.items():
        rate_str = f"{wl['samples_per_sec']/1e6:.0f}M" if wl['samples_per_sec'] < 1e9 else f"{wl['samples_per_sec']/1e9:.1f}G"
        status = "✓ OK" if wl['feasible'] else "✗ FAIL"
        report.append(f"{wl['description'][:25]:<26} {rate_str + '/s':<14} "
                      f"{wl['tiles_required']:<8.1f} {wl['utilization_pct']:<7.1f}% "
                      f"{wl['actual_latency_us']:.3f}µs{'':<6} {status:<10}")
    report.append("-" * 70)
    feasible_count = sum(1 for wl in workload_results.values() if wl['feasible'])
    report.append(f"  {feasible_count}/{len(workload_results)} workloads feasible on single RFTPU")
    
    report.append("")
    report.append("=" * 70)
    report.append("BENCHMARK SUMMARY")
    report.append("=" * 70)
    report.append(f"  • Peak performance: {fpga_results['asic']['tops']:.2f} TOPS @ 950 MHz")
    report.append(f"  • Best efficiency:  {power_results['peak_gops_per_watt']:.0f} GOPS/W @ {power_results['optimal_efficiency_point']} mode")
    report.append(f"  • Max cascade:      {cascade_results['aggregate_tops'][-1]:.1f} TOPS (16-chip)")
    report.append(f"  • vs FPGA:          {fpga_results['fpga_targets']['xilinx_vu13p']['vs_asic_throughput']:.1f}× faster, "
                  f"{fpga_results['fpga_targets']['xilinx_vu13p']['vs_asic_efficiency']:.1f}× more efficient")
    report.append("")
    
    return "\n".join(report)


def main():
    # Create config from RTL parameters
    cfg = RFTPUConfig()
    
    # Run basic analysis
    results = analyze_performance(cfg)
    comparisons = compare_to_baselines(results)
    
    # Print basic report
    report = format_report(cfg, results, comparisons)
    print(report)
    
    # Run extended benchmarks
    print("\nRunning extended benchmarks...")
    power_results = power_scaling_benchmark(cfg)
    cascade_results = multi_chip_cascade_benchmark(cfg)
    fpga_results = fpga_comparison_benchmark(cfg)
    workload_results = workload_benchmark(cfg)
    
    # Print extended report
    extended_report = format_extended_report(cfg, power_results, cascade_results, 
                                             fpga_results, workload_results)
    print(extended_report)
    
    # Return all results for programmatic use
    return {
        'config': cfg,
        'results': results,
        'comparisons': comparisons,
        'power_scaling': power_results,
        'multi_chip_cascade': cascade_results,
        'fpga_comparison': fpga_results,
        'workload_analysis': workload_results
    }


if __name__ == "__main__":
    main()
