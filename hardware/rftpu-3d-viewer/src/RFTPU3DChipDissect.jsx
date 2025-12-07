import React, { useRef, useState, useEffect, useMemo, useCallback } from 'react';
import * as THREE from 'three';
import { EffectComposer } from 'three/examples/jsm/postprocessing/EffectComposer';
import { RenderPass } from 'three/examples/jsm/postprocessing/RenderPass';
import { UnrealBloomPass } from 'three/examples/jsm/postprocessing/UnrealBloomPass';
import { SMAAPass } from 'three/examples/jsm/postprocessing/SMAAPass';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls';

// ===============================================
// ANIMATION & EASING UTILITIES
// ===============================================
const SPRING_CONFIG = { tension: 120, friction: 14 };
const EASE_OUT_EXPO = t => (t === 1 ? 1 : 1 - Math.pow(2, -10 * t));
const EASE_IN_OUT_CUBIC = t => t < 0.5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2;
const lerp = (start, end, factor) => start + (end - start) * factor;
const smoothDamp = (current, target, velocity, smoothTime, deltaTime) => {
  const omega = 2 / Math.max(0.0001, smoothTime);
  const x = omega * deltaTime;
  const exp = 1 / (1 + x + 0.48 * x * x + 0.235 * x * x * x);
  const change = current - target;
  const temp = (velocity + omega * change) * deltaTime;
  const newVelocity = (velocity - omega * temp) * exp;
  const newValue = target + (change + temp) * exp;
  return { value: newValue, velocity: newVelocity };
};

// ===============================================
// RFT-PROCESSOR (Resonance Fourier Transform Processor)
// Interactive 3D Hardware Visualization
// Based on quantoniumos_unified_engines.sv & rftpu_architecture.tlv
// USPTO Patent Application #19/169,399 • DOI: 10.5281/zenodo.17822056
// Copyright (C) 2025 Luis M. Minier / QuantoniumOS
// ===============================================

const TILE_DIM = 8;  // From rftpu_pkg::TILE_DIM
const TILE_COUNT = TILE_DIM * TILE_DIM;  // 64 tiles
const SAMPLE_WIDTH = 16;  // From rftpu_pkg::SAMPLE_WIDTH
const BLOCK_SAMPLES = 8;  // From rftpu_pkg::BLOCK_SAMPLES
const DIGEST_WIDTH = 256;  // From rftpu_pkg::DIGEST_WIDTH
const SCRATCH_DEPTH = 64;  // From rftpu_pkg::SCRATCH_DEPTH
const TOPO_MEM_DEPTH = 64;  // From rftpu_pkg::TOPO_MEM_DEPTH
const MAX_INFLIGHT = 64;  // From rftpu_pkg::MAX_INFLIGHT
const HOP_LATENCY = 2;  // From rftpu_pkg::HOP_LATENCY

// PHI constant from canonical_rft_core: 0x0001_9E37 in Q16.16 = 1.618034
const PHI_Q16_16 = 0x00019E37;
const PHI_DECIMAL = 1.618033988749895;

// ===============================================
// BENCHMARK METRICS FROM PAPER (Table 2 & 3)
// ===============================================
const BENCHMARK_METRICS = {
  peakPerformance: { value: 2.39, unit: 'TOPS', label: 'Peak Performance' },
  efficiency: { value: 291, unit: 'GOPS/W', label: 'Energy Efficiency' },
  powerDissipation: { value: 8.2, unit: 'W', label: 'Power Dissipation' },
  tileUtilization: { value: 94.2, unit: '%', label: 'Tile Utilization' },
  nocBandwidth: { value: 460, unit: 'GB/s', label: 'NoC Bandwidth' },
  sisLatency: { value: 142, unit: 'cycles', label: 'SIS Latency' },
  feistelThroughput: { value: 3.2, unit: 'Gb/s', label: 'Feistel Throughput' }
};

// ===============================================
// BENCHMARK SCENARIOS FROM PAPER
// ===============================================
const BENCHMARK_SCENARIOS = {
  signalProcessing: {
    id: 'rft',
    name: '1024-point RFT Transform',
    tiles: 16,
    duration: 12.5,
    durationUnit: 'μs',
    throughput: '2.39 TOPS',
    description: 'Resonant Fourier Transform with φ-basis functions'
  },
  cascadeH3: {
    id: 'h3',
    name: 'H³ Graph Traversal',
    tiles: 64,
    nocHops: 240,
    latency: '480 cycles',
    description: 'Multi-hop cascade with dual-slot vertex routing'
  },
  sisHash: {
    id: 'sis',
    name: 'SIS-512 Lattice Hash',
    blocks: 8,
    latency: '142 cycles',
    security: 'Post-quantum',
    description: 'N=512 lattice-based cryptographic hash'
  },
  feistelEncrypt: {
    id: 'feistel',
    name: 'Feistel-48 Cipher',
    rounds: 48,
    throughput: '3.2 Gb/s',
    description: '48-round Feistel network encryption'
  }
};

// ===============================================
// THERMAL PROFILING (from PHYSICAL_DESIGN_SPEC.md)
// ===============================================
const THERMAL_PROFILE = {
  idle: { temp: 45, color: 0x0066ff, label: 'Idle (45°C)' },
  light: { temp: 65, color: 0x00ff66, label: 'Light (65°C)' },
  active: { temp: 85, color: 0xff6600, label: 'Active (85°C)' },
  peak: { temp: 105, color: 0xff0000, label: 'Peak (105°C)' }
};

// ===============================================
// POWER DOMAINS (from PHYSICAL_DESIGN_SPEC.md)
// ===============================================
const POWER_DOMAINS = {
  VDD_TILE: { voltage: 0.75, power: 5440, unit: 'mW', color: 0x00d4aa, label: '64 Tiles' },
  VDD_NOC: { voltage: 0.80, power: 620, unit: 'mW', color: 0x4a9eff, label: 'NoC Fabric' },
  VDD_SIS: { voltage: 0.75, power: 1100, unit: 'mW', color: 0xff6b9d, label: 'SIS Hash' },
  VDD_FEISTEL: { voltage: 0.70, power: 280, unit: 'mW', color: 0xffd700, label: 'Feistel-48' }
};

// ===============================================
// ARCHITECTURE COMPARISON (Table 4 from paper)
// ===============================================
const COMPARISON_DATA = {
  rftpu: { name: 'RFT-Processor', tops: 2.39, efficiency: 291, area: 72.25, process: '7nm' },
  traditionalFFT: { name: 'Traditional FFT', tops: 1.8, efficiency: 185, area: 85, process: '7nm' },
  dspAccel: { name: 'DSP Accelerator', tops: 1.5, efficiency: 142, area: 95, process: '7nm' },
  advantage: { ops: '+32%', efficiency: '+57%', area: '-15%' }
};

// ===============================================
// RFT OPERATION VISUALIZATION
// ===============================================
const RFT_VISUALIZATION = {
  kernelCoeffs: '8×8 Q1.15 matrix',
  phiRotations: 'φ = 1.618034 golden ratio phases',
  cordicIterations: '16-iteration sin/cos',
  outputDigest: '256-bit SIS digest',
  pipelineStages: ['Sample In', 'CORDIC', 'Kernel Mult', 'Accumulate', 'Digest Out']
};

// ===============================================
// PAPER REFERENCES
// ===============================================
const PAPER_REFERENCES = {
  doi: '10.5281/zenodo.17822056',
  patent: 'USPTO #19/169,399',
  architecture: 'Fig. 2: RFTPU Architecture Overview',
  floorplan: 'Fig. 3: Physical Design Floorplan',
  performance: 'Table 2: Benchmark Results (2.39 TOPS)',
  efficiency: 'Table 3: Power Analysis (291 GOPS/W)',
  comparison: 'Table 4: Comparison vs. State-of-Art'
};

// Clock domains from PHYSICAL_DESIGN_SPEC.md
const CLOCK_DOMAINS = {
  clk_tile: { freq: 950, color: 0x00d4aa, name: 'clk_tile (950 MHz)' },
  clk_noc: { freq: 1200, color: 0x4a9eff, name: 'clk_noc (1.2 GHz)' },
  clk_sis: { freq: 475, color: 0xff6b9d, name: 'clk_sis (475 MHz)' },
  clk_feistel: { freq: 1400, color: 0xffd700, name: 'clk_feistel (1.4 GHz)' }
};

// Diagonal tile activation pattern (thermal optimization from spec)
const ACTIVATION_WAVES = [
  [0, 9, 18, 27, 36, 45, 54, 63],  // Wave 1
  [1, 10, 19, 28, 37, 46, 55],     // Wave 2
  [2, 11, 20, 29, 38, 47, 56],     // Wave 3
  [3, 12, 21, 30, 39, 48, 57],     // Wave 4
  [4, 13, 22, 31, 40, 49, 58],     // Wave 5
  [5, 14, 23, 32, 41, 50, 59],     // Wave 6
  [6, 15, 24, 33, 42, 51, 60],     // Wave 7
  [7, 16, 25, 34, 43, 52, 61],     // Wave 8
  [8, 17, 26, 35, 44, 53, 62],     // Wave 9
];

const EXPLANATIONS = {
  heatspreader: {
    name: "Integrated Heat Spreader (IHS)",
    simple: "The nickel-plated copper lid that sits on top of the RFT-Processor die. It pulls heat away from the compute tiles and spreads it across the cooling solution. Essential for maintaining junction temperature under 105°C.",
    technical: "Thermal interface between die and cooling solution. Designed for 8.5°C/W package θJA with CoWoS active cooling. Hotspot budget: <150 mW/mm² per 0.5mm² region.",
    specs: "Material: Cu-Ni • θJA: 8.5°C/W • TDP: <9W peak • Tj_max: 105°C"
  },
  die: {
    name: "RFT-Processor Compute Die",
    simple: "The silicon die containing 64 processing tiles in an 8×8 mesh. Built on TSMC 7nm FinFET process. This is where all Resonant Fourier Transform computations happen.",
    technical: "Monolithic silicon substrate with 64 rpu_tile_shell instances, rpu_noc_fabric, quantoniumos_unified_core controller. Total ~3.2M gates, ~2.8mm² active area.",
    specs: "Process: TSMC N7FF (7nm FinFET) • Die: 8.5×8.5mm • Gates: 3.2M • SRAM: 640Kb"
  },
  tile: {
    name: "RPU Tile Shell (rpu_tile_shell)",
    simple: "One of 64 processing tiles. Each contains a phi_rft_core for RFT computation, 64-entry scratchpad SRAM, 64-entry topology memory for H³ graph operations, and a wormhole NoC router.",
    technical: "Module rpu_tile_shell: FSM controller + phi_rft_core + scratch[64] + topo_mem[64] + router interface. Supports modes: RFT transform, cascade, H³ multicast.",
    specs: "Gates: 32,000 • Area: 12,800μm² • Power: 85mW active / 4mW idle • Freq: 950MHz (clk_tile)"
  },
  phirft: {
    name: "Φ-RFT Core (phi_rft_core)",
    simple: "The Resonant Fourier Transform engine inside each tile. Uses the golden ratio φ=1.618034 to generate unique basis functions. Processes 8 samples per block with 12-cycle latency.",
    technical: "Module phi_rft_core: Implements Ψ = Σ_i w_i D_φi C_σi D†_φi with CORDIC-based sin/cos. Kernel ROM stores 8×8 Q1.15 coefficients. Outputs 256-bit SIS digest + resonance_flag.",
    specs: "Latency: 12 cycles • Samples: 8/block • φ = 0x9E37 (Q16.16) • Digest: 256-bit • Gates: 24,000"
  },
  scratchpad: {
    name: "Scratchpad SRAM (scratch[64])",
    simple: "64-entry local memory buffer for intermediate RFT coefficients. Each tile has its own scratchpad for fast access without going through the NoC.",
    technical: "64-entry × 16-bit SRAM macro for intermediate coefficients and working data. Power-gated during idle. Single-cycle access latency.",
    specs: "Depth: 64 entries • Width: 16-bit • Size: 2Kb • Latency: 1 cycle • Total: 128×2Kb = 256Kb"
  },
  topo: {
    name: "Topology Memory (topo_mem[64])",
    simple: "64-entry memory that stores H³ graph vertex connections. Used for cascade operations where tiles send results to specific neighbor vertices.",
    technical: "Module topo_mem: 64-entry H³ vertex accumulator. Stores vertex_id[5:0], dest_x[2:0], dest_y[2:0] for cascade routing. Supports dual-slot H³ multicast.",
    specs: "Depth: 64 entries • Width: 4Kb • Vertices: 64 • Update: 950MHz • Total: 64×4Kb = 256Kb"
  },
  router: {
    name: "Wormhole NoC Router",
    simple: "Each tile's connection to the Network-on-Chip. Routes 256-bit digest packets between tiles using X-Y dimension-order routing with 2-cycle hop latency.",
    technical: "5-port wormhole router (N/S/E/W/Local). Supports noc_payload_t: digest[256] + vertex_id[6] + mode[4] + src/dest[6] + pkt_type[2]. Inflight capacity: 64 packets.",
    specs: "Ports: 5 • Latency: 2 cycles/hop • Buffers: 4 flits • Payload: 280 bits • Freq: 1.2GHz (clk_noc)"
  },
  noc: {
    name: "RPU NoC Fabric (rpu_noc_fabric)",
    simple: "The 8×8 mesh network connecting all 64 tiles. Data packets travel along horizontal and vertical links, hopping from router to router in 2 cycles per hop.",
    technical: "Module rpu_noc_fabric: 64 routers in 8×8 mesh topology. Wormhole routing with 64 inflight slots. Supports digest packets, data packets, and control packets.",
    specs: "Topology: 8×8 Mesh • Diameter: 14 hops • Gates: 95,000 • Freq: 1.2GHz • BW: ~460GB/s aggregate"
  },
  bondwires: {
    name: "Wire Bonds (Signal + Power)",
    simple: "Gold wire bonds connecting the die pads to the package substrate. Carries signals to/from the host interface and power/ground connections.",
    technical: "Gold wire bonds for signal and power connectivity. Supports PCIe Gen5 x8 (16 diff pairs), JTAG, SPI, and cascade LVDS links.",
    specs: "Material: 99.99% Au • Diameter: 25μm • Signal Pins: ~420 • Power/Ground: ~380 • Total: ~800"
  },
  bga: {
    name: "BGA-800 Solder Balls",
    simple: "The chip's connection to the PCB - an array of solder balls on the bottom of the package. Uses 0.8mm pitch for the 25×25mm flip-chip package.",
    technical: "Ball Grid Array providing mechanical attachment and electrical interface. Designed for <3% IR drop with 380 power/ground balls.",
    specs: "Pitch: 0.8mm • Material: SAC305 • Count: ~800 balls • Package: 25×25mm flip-chip"
  },
  substrate: {
    name: "Package Substrate (CoWoS-S)",
    simple: "2.5D silicon interposer with organic substrate. Routes signals between the die, HBM2E memory, and BGA balls. Uses TSMC CoWoS-S technology.",
    technical: "12-layer organic substrate (8 thin + 3 thick + 1 RDL). Silicon interposer (100μm, 65nm RDL) with 40μm TSV pitch. Supports HBM2E integration.",
    specs: "Layers: 12 • Interposer: 100μm Si • TSV: 40μm pitch • Material: BT resin • HBM: 2×8GB"
  },
  cordic: {
    name: "CORDIC Sin/Cos Engine (cordic_sincos)",
    simple: "The trigonometry calculator inside each RFT core. Uses 16 iterations of the CORDIC algorithm to compute sin/cos values for the RFT basis functions.",
    technical: "Module cordic_sincos: 16-iteration CORDIC for sin/cos generation. Also cordic_cartesian_to_polar (12-iter) for magnitude/phase extraction.",
    specs: "Iterations: 16 • Width: 32-bit • Gates: 4,200 • Latency: 16 cycles • Precision: Q16.16"
  },
  sis: {
    name: "SIS Hash Engine (rft_sis_hash_v31)",
    simple: "The security engine that generates lattice-based hash digests. Uses N=512 expansion and matrix operations for post-quantum cryptographic security.",
    technical: "Module rft_sis_hash_v31: N=512 SIS expansion + lattice math. Includes 2×256Kb A-matrix SRAM cache. Runs on clk_sis at 475MHz.",
    specs: "N: 512 • Digest: 256-bit • Gates: 320,000 • Power: 1,100mW • Freq: 475MHz (clk_sis)"
  },
  feistel: {
    name: "Feistel-48 Cipher (feistel_48_cipher)",
    simple: "A 48-round Feistel cipher for stream encryption. Runs at high frequency for low-latency cryptographic operations.",
    technical: "Module feistel_48_cipher: 48-round Feistel network with feistel_round_function. Combinational F-function, sequential rounds controller.",
    specs: "Rounds: 48 • Gates: 45,000 • F-function: 8,500 gates • Power: 280mW • Freq: 1.4GHz (clk_feistel)"
  },
  qlogo: {
    name: "QuantoniumOS / RFT-Processor Logo",
    simple: "The 'Q' represents QuantoniumOS - the hybrid computational framework implementing the Resonant Fourier Transform. Created by Luis M. Minier.",
    technical: "Brand identifier for USPTO Patent Application #19/169,399 - Hybrid Computational Framework for Quantum and Resonance Simulation using Resonant Fourier Transform.",
    specs: "Patent: USPTO #19/169,399 • Filed: 2024 • Inventor: Luis M. Minier • Framework: QuantoniumOS"
  },
  dma: {
    name: "DMA Ingress Controller (rpu_dma_ingress)",
    simple: "The data loading engine that distributes sample data to all 64 tiles. Demultiplexes incoming streams to the correct tile based on tile_id.",
    technical: "Module rpu_dma_ingress: 64-tile demux for dma_frame_t packets. Connects to HBM2E PHY (4 channels, 460GB/s) for sample data streaming.",
    specs: "Gates: 6,500 • Payload: samples[128] + block_idx[16] + tile_id[7] • BW: 460GB/s (HBM2E)"
  },
  controller: {
    name: "Unified Controller (quantoniumos_unified_core)",
    simple: "The master controller that orchestrates all engines: RFT tiles, SIS hash, Feistel cipher, and DMA. Runs the top-level FSM and pipeline scheduling.",
    technical: "Module quantoniumos_unified_core: Top controller + pipeline FSM. Manages tile activation patterns, clock domain crossings, and engine coordination.",
    specs: "Gates: 420,000 • Power: 180mW active / 12mW idle • Clocks: tile/noc/sis/feistel domains"
  },
  unified_controller: {
    name: "Unified Controller (quantoniumos_unified_core)",
    simple: "The master controller that orchestrates all engines: RFT tiles, SIS hash, Feistel cipher, and DMA. Runs the top-level FSM and pipeline scheduling.",
    technical: "Module quantoniumos_unified_core: Top controller + pipeline FSM. Manages tile activation patterns, clock domain crossings, and engine coordination.",
    specs: "Gates: 420,000 • Power: 180mW active / 12mW idle • Location: Top of SPINE (right edge)"
  },
  sis_hash: {
    name: "SIS Hash Engine (rft_sis_hash_v31)",
    simple: "The security engine that generates lattice-based hash digests. Uses N=512 expansion and matrix operations for post-quantum cryptographic security.",
    technical: "Module rft_sis_hash_v31: N=512 SIS expansion + lattice math. Includes 2×256Kb A-matrix SRAM cache. Runs on clk_sis at 475MHz.",
    specs: "N: 512 • Digest: 256-bit • Gates: 320,000 • Power: 1,100mW • Freq: 475MHz • Location: Center of SPINE"
  },
  pll: {
    name: "Phase-Locked Loop (3.8 GHz)",
    simple: "Clock generation circuits in the NW and SE corners of the die. Generate all 4 clock domains: tile (950MHz), noc (1.2GHz), sis (475MHz), and feistel (1.4GHz).",
    technical: "TSMC analog IP PLL with 3.8GHz VCO. Clock dividers generate clk_tile (÷4), clk_noc (÷3.17), clk_sis (÷8), clk_feistel (÷2.7). Isolated analog islands.",
    specs: "VCO: 3.8 GHz • Domains: 4 • Jitter: <5ps RMS • Island Size: 0.5mm × 0.5mm each • Location: NW + SE corners"
  },
  dma_ingress: {
    name: "DMA Ingress Controller (rpu_dma_ingress)",
    simple: "The data loading engine that distributes sample data to all 64 tiles. Demultiplexes incoming streams to the correct tile based on tile_id.",
    technical: "Module rpu_dma_ingress: 64-tile demux for dma_frame_t packets. Connects to HBM2E PHY (4 channels, 460GB/s) for sample data streaming.",
    specs: "Gates: 6,500 • Payload: samples[128] + block_idx[16] + tile_id[7] • BW: 460GB/s • Location: South edge (full width)"
  }
};

const TOUR_STOPS = [
  { component: 'heatspreader', explode: 0, rotation: { x: -0.3, y: 0.4 }, zoom: 1.0, duration: 3000 },
  { component: 'heatspreader', explode: 2, rotation: { x: -0.3, y: 0.4 }, zoom: 1.0, duration: 2000 },
  { component: 'qlogo', explode: 2, rotation: { x: -0.5, y: 0 }, zoom: 1.5, duration: 3000 },
  { component: 'die', explode: 1.5, rotation: { x: -0.6, y: 0.5 }, zoom: 1.3, duration: 3000 },
  { component: 'tile', explode: 1.5, rotation: { x: -0.8, y: 0.3 }, zoom: 1.8, duration: 3000 },
  { component: 'phirft', explode: 1.5, rotation: { x: -0.8, y: 0.3 }, zoom: 2.2, duration: 3000 },
  { component: 'scratchpad', explode: 1.5, rotation: { x: -0.8, y: 0.5 }, zoom: 2.2, duration: 2500 },
  { component: 'topo', explode: 1.5, rotation: { x: -0.8, y: 0.6 }, zoom: 2.2, duration: 2500 },
  { component: 'router', explode: 1.5, rotation: { x: -0.8, y: 0.7 }, zoom: 2.0, duration: 2500 },
  { component: 'noc', explode: 1.0, rotation: { x: -0.6, y: 0.7 }, zoom: 1.4, duration: 3000 },
  { component: 'bondwires', explode: 1.2, rotation: { x: -0.2, y: 1.8 }, zoom: 1.4, duration: 2500 },
  { component: 'substrate', explode: 1.5, rotation: { x: -0.1, y: 2.0 }, zoom: 1.3, duration: 2500 },
  { component: 'bga', explode: 2, rotation: { x: 0.1, y: 2.4 }, zoom: 1.5, duration: 2500 },
];

export default function RFTPU3DChipDissect() {
  const containerRef = useRef(null);
  const rendererRef = useRef(null);
  const composerRef = useRef(null);
  const sceneRef = useRef(null);
  const cameraRef = useRef(null);
  const layersRef = useRef({});
  const raycasterRef = useRef(new THREE.Raycaster());
  const mouseRef = useRef(new THREE.Vector2());
  
  const [rotation, setRotation] = useState({ x: -0.35, y: 0.4 });
  const [targetRotation, setTargetRotation] = useState(null);
  const [zoom, setZoom] = useState(1);
  const [targetZoom, setTargetZoom] = useState(null);
  const [isDragging, setIsDragging] = useState(false);
  const [lastMouse, setLastMouse] = useState({ x: 0, y: 0 });
  const [hoveredComponent, setHoveredComponent] = useState(null);
  const [selectedComponent, setSelectedComponent] = useState(null);
  const [explodeLevel, setExplodeLevel] = useState(0);
  const [targetExplode, setTargetExplode] = useState(null);
  
  // Enhanced UX states
  const [isLoading, setIsLoading] = useState(true);
  const [showOnboarding, setShowOnboarding] = useState(true);
  const [activePanel, setActivePanel] = useState('controls'); // controls, components, benchmarks
  const [uiMinimized, setUiMinimized] = useState(false);
  const [transitionProgress, setTransitionProgress] = useState(1);
  
  // Smooth animation velocities for spring physics
  const velocityRef = useRef({ rotX: 0, rotY: 0, zoom: 0, explode: 0 });
  const orbitControlsRef = useRef(null);
  const [showLayers, setShowLayers] = useState({
    heatspreader: true,
    die: true,
    tiles: true,
    noc: true,
    spine: true,      // SPINE: SIS + Feistel + Unified Controller
    dma: true,        // DMA Ingress (south edge)
    pll: true,        // PLL islands (NW/SE corners)
    bondwires: true,
    bga: true,
    substrate: true,
    qlogo: true
  });
  const [animateNoc, setAnimateNoc] = useState(true);
  const [simpleMode, setSimpleMode] = useState(true);
  const [visualMode, setVisualMode] = useState('normal'); // normal, xray, thermal, dataflow
  const [tourActive, setTourActive] = useState(false);
  const [tourStep, setTourStep] = useState(0);
  const [showMetrics, setShowMetrics] = useState(true);
  const [autoRotate, setAutoRotate] = useState(false);
  const [showClockDomains, setShowClockDomains] = useState(false);
  const [showTileLabels, setShowTileLabels] = useState(true);
  const [activeTile, setActiveTile] = useState(null);
  const [showDataflow, setShowDataflow] = useState(false);
  const [performanceMode, setPerformanceMode] = useState('idle'); // idle, active, thermal
  const [highlightWave, setHighlightWave] = useState(-1);
  
  // Benchmark & Visualization State
  const [benchmarkMode, setBenchmarkMode] = useState(null); // null, 'rft', 'h3', 'sis', 'feistel'
  const [benchmarkRunning, setBenchmarkRunning] = useState(false);
  const [benchmarkProgress, setBenchmarkProgress] = useState(0);
  const [showBenchmarkPanel, setShowBenchmarkPanel] = useState(false);
  const [showThermalView, setShowThermalView] = useState(false);
  const [thermalWaveIndex, setThermalWaveIndex] = useState(0);
  const [showPowerDomains, setShowPowerDomains] = useState(false);
  const [showComparison, setShowComparison] = useState(false);
  const [liveMetrics, setLiveMetrics] = useState({
    currentTOPS: 0,
    currentPower: 0.54,
    activeTiles: 0,
    nocUtilization: 0
  });
  const [rftStage, setRftStage] = useState(0); // 0-4 for pipeline stages
  
  const frameRef = useRef(0);
  const tourTimeoutRef = useRef(null);
  const tileLabelsRef = useRef([]);
  const benchmarkIntervalRef = useRef(null);
  const thermalIntervalRef = useRef(null);

  // Enhanced smooth interpolation with spring physics
  useEffect(() => {
    const deltaTime = 0.016; // ~60fps
    const smoothTime = 0.25; // Smoother transitions
    
    const interval = setInterval(() => {
      if (targetRotation) {
        // Spring-based rotation interpolation
        const dampX = smoothDamp(rotation.x, targetRotation.x, velocityRef.current.rotX, smoothTime, deltaTime);
        const dampY = smoothDamp(rotation.y, targetRotation.y, velocityRef.current.rotY, smoothTime, deltaTime);
        velocityRef.current.rotX = dampX.velocity;
        velocityRef.current.rotY = dampY.velocity;
        setRotation({ x: dampX.value, y: dampY.value });
        
        if (Math.abs(rotation.x - targetRotation.x) < 0.005 && Math.abs(rotation.y - targetRotation.y) < 0.005) {
          setTargetRotation(null);
          velocityRef.current.rotX = 0;
          velocityRef.current.rotY = 0;
        }
      }
      if (targetZoom !== null) {
        const dampZoom = smoothDamp(zoom, targetZoom, velocityRef.current.zoom, smoothTime, deltaTime);
        velocityRef.current.zoom = dampZoom.velocity;
        setZoom(dampZoom.value);
        if (Math.abs(zoom - targetZoom) < 0.005) {
          setTargetZoom(null);
          velocityRef.current.zoom = 0;
        }
      }
      if (targetExplode !== null) {
        const dampExplode = smoothDamp(explodeLevel, targetExplode, velocityRef.current.explode, smoothTime * 1.5, deltaTime);
        velocityRef.current.explode = dampExplode.velocity;
        setExplodeLevel(dampExplode.value);
        if (Math.abs(explodeLevel - targetExplode) < 0.005) {
          setTargetExplode(null);
          velocityRef.current.explode = 0;
        }
      }
    }, 16);
    return () => clearInterval(interval);
  }, [targetRotation, rotation, targetZoom, zoom, targetExplode, explodeLevel]);

  // Tour system
  useEffect(() => {
    if (!tourActive) return;
    
    if (tourStep >= TOUR_STOPS.length) {
      setTourActive(false);
      setTourStep(0);
      return;
    }

    const stop = TOUR_STOPS[tourStep];
    setSelectedComponent(stop.component);
    setTargetExplode(stop.explode);
    setTargetRotation(stop.rotation);
    setTargetZoom(stop.zoom);

    tourTimeoutRef.current = setTimeout(() => {
      setTourStep(prev => prev + 1);
    }, stop.duration);

    return () => {
      if (tourTimeoutRef.current) clearTimeout(tourTimeoutRef.current);
    };
  }, [tourActive, tourStep]);

  // Initialize Three.js scene with post-processing
  useEffect(() => {
    if (!containerRef.current) return;

    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0x0a0e14);
    scene.fog = new THREE.Fog(0x0a0e14, 20, 50);
    sceneRef.current = scene;

    const camera = new THREE.PerspectiveCamera(45, containerRef.current.clientWidth / containerRef.current.clientHeight, 0.1, 1000);
    camera.position.set(0, 8, 12);
    camera.lookAt(0, 0, 0);
    cameraRef.current = camera;

    const renderer = new THREE.WebGLRenderer({ 
      antialias: true,
      alpha: true,
      powerPreference: "high-performance"
    });
    renderer.setSize(containerRef.current.clientWidth, containerRef.current.clientHeight);
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    renderer.shadowMap.enabled = true;
    renderer.shadowMap.type = THREE.PCFSoftShadowMap;
    renderer.toneMapping = THREE.ACESFilmicToneMapping;
    renderer.toneMappingExposure = 1.2;
    containerRef.current.appendChild(renderer.domElement);
    rendererRef.current = renderer;

    // Post-processing
    const composer = new EffectComposer(renderer);
    composer.addPass(new RenderPass(scene, camera));
    
    const bloomPass = new UnrealBloomPass(
      new THREE.Vector2(window.innerWidth, window.innerHeight),
      0.6, // strength
      0.4, // radius
      0.85 // threshold
    );
    composer.addPass(bloomPass);
    composerRef.current = composer;

    // Enhanced lighting
    const ambientLight = new THREE.AmbientLight(0xffffff, 0.3);
    scene.add(ambientLight);

    const mainLight = new THREE.DirectionalLight(0xffffff, 1.2);
    mainLight.position.set(10, 20, 10);
    mainLight.castShadow = true;
    mainLight.shadow.mapSize.width = 2048;
    mainLight.shadow.mapSize.height = 2048;
    scene.add(mainLight);

    const fillLight = new THREE.DirectionalLight(0x4a9eff, 0.4);
    fillLight.position.set(-10, 10, -10);
    scene.add(fillLight);

    const rimLight = new THREE.DirectionalLight(0x00d4aa, 0.3);
    rimLight.position.set(0, -10, 10);
    scene.add(rimLight);

    // Accent lights for drama
    const accent1 = new THREE.PointLight(0x00d4aa, 0.5, 20);
    accent1.position.set(5, 5, 5);
    scene.add(accent1);

    const accent2 = new THREE.PointLight(0x4a9eff, 0.5, 20);
    accent2.position.set(-5, 5, -5);
    scene.add(accent2);

    // Create chip layers matching PHYSICAL_DESIGN_SPEC.md floorplan
    const layers = {
      substrate: createSubstrate(scene),
      bga: createBGA(scene),
      bondwires: createBondWires(scene),
      die: createDie(scene),
      tiles: createTileArray(scene),
      noc: createNoC(scene),
      spine: createSpineModules(scene),  // SIS + Feistel + Unified Controller
      dma: createDMAIngress(scene),       // DMA Ingress on south edge
      pll: createPLLIslands(scene),       // PLL_NW and PLL_SE
      heatspreader: createHeatspreader(scene),
      qlogo: createQLogo(scene)
    };
    layersRef.current = layers;

    // Animation loop
    let animationId;
    const animate = () => {
      animationId = requestAnimationFrame(animate);
      frameRef.current += 1;
      
      // Auto-rotate
      if (autoRotate && !isDragging && !tourActive) {
        setRotation(prev => ({ ...prev, y: prev.y + 0.002 }));
      }

      // Animate NoC particles with advanced patterns
      if (animateNoc && layers.noc.userData.particles) {
        layers.noc.userData.particles.forEach((particle, i) => {
          const t = (frameRef.current * 0.015 + i * 0.1) % 1;
          const path = particle.userData.path;
          
          // Smooth easing
          const easeT = t < 0.5 ? 2 * t * t : 1 - Math.pow(-2 * t + 2, 2) / 2;
          particle.position.lerpVectors(path.start, path.end, easeT);
          
          // Pulse effect
          const scale = 0.8 + Math.sin(t * Math.PI) * 0.4;
          particle.scale.setScalar(scale);
          
          // Color shift based on position
          const intensity = 0.6 + Math.sin(t * Math.PI * 2) * 0.4;
          particle.material.emissiveIntensity = intensity;
        });
      }

      // Animate Q logo
      if (layers.qlogo.userData.glowMesh) {
        const pulse = Math.sin(frameRef.current * 0.03) * 0.3 + 0.7;
        layers.qlogo.userData.glowMesh.material.opacity = pulse * 0.5;
        layers.qlogo.userData.glowMesh.material.emissiveIntensity = pulse * 0.6;
      }

      // Pulsing glow on selected tiles
      if (selectedComponent && layers.tiles) {
        layers.tiles.children.forEach(child => {
          if (child.userData.component === selectedComponent) {
            const pulse = Math.sin(frameRef.current * 0.1) * 0.3 + 0.7;
            if (child.material.emissive) {
              child.material.emissiveIntensity = pulse * 0.3;
            }
          }
        });
      }

      // Thermal visualization mode
      if (visualMode === 'thermal') {
        layers.tiles?.children.forEach((child, i) => {
          if (child.material && child.material.color) {
            const heat = (Math.sin(frameRef.current * 0.02 + i * 0.5) + 1) / 2;
            child.material.color.setHSL(0.6 - heat * 0.6, 0.8, 0.5);
            child.material.emissive.setHSL(0.6 - heat * 0.6, 1, heat * 0.3);
          }
        });
      }
      
      composer.render();
    };
    animate();

    // Initialize OrbitControls for smooth camera manipulation
    const controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.dampingFactor = 0.08;
    controls.rotateSpeed = 0.5;
    controls.zoomSpeed = 0.8;
    controls.panSpeed = 0.5;
    controls.minDistance = 5;
    controls.maxDistance = 40;
    controls.maxPolarAngle = Math.PI * 0.85;
    controls.minPolarAngle = Math.PI * 0.1;
    controls.target.set(0, 0, 0);
    controls.enabled = true;
    orbitControlsRef.current = controls;

    const handleResize = () => {
      if (!containerRef.current) return;
      camera.aspect = containerRef.current.clientWidth / containerRef.current.clientHeight;
      camera.updateProjectionMatrix();
      renderer.setSize(containerRef.current.clientWidth, containerRef.current.clientHeight);
      composer.setSize(containerRef.current.clientWidth, containerRef.current.clientHeight);
    };
    window.addEventListener('resize', handleResize);
    
    // Simulate loading completion
    setTimeout(() => setIsLoading(false), 1200);

    return () => {
      window.removeEventListener('resize', handleResize);
      cancelAnimationFrame(animationId);
      controls.dispose();
      if (containerRef.current && renderer.domElement) {
        containerRef.current.removeChild(renderer.domElement);
      }
      renderer.dispose();
      composer.dispose();
    };
  }, [animateNoc, isDragging, autoRotate, tourActive, selectedComponent, visualMode]);

  // Update layer visibility, materials and explode positions
  useEffect(() => {
    if (!layersRef.current) return;
    
    const explodeOffsets = {
      heatspreader: explodeLevel * 2.5,
      qlogo: explodeLevel * 2.6,
      tiles: explodeLevel * 0.8,
      noc: explodeLevel * 1.0,
      die: explodeLevel * 0.3,
      bondwires: 0,
      substrate: -explodeLevel * 0.5,
      bga: -explodeLevel * 1.5
    };

    Object.entries(layersRef.current).forEach(([key, group]) => {
      group.visible = showLayers[key] !== false;
      group.position.y = explodeOffsets[key] || 0;
      
      // X-ray mode
      group.children.forEach(child => {
        if (child.material) {
          if (visualMode === 'xray') {
            child.material.transparent = true;
            child.material.opacity = 0.3;
            child.material.wireframe = false;
          } else {
            child.material.transparent = child.userData.originalTransparent || false;
            child.material.opacity = child.userData.originalOpacity || 1;
            child.material.wireframe = false;
          }
        }
      });
    });
  }, [showLayers, explodeLevel, visualMode]);

  // Update camera
  useEffect(() => {
    if (!cameraRef.current) return;
    const radius = 15 / zoom;
    cameraRef.current.position.x = radius * Math.sin(rotation.y) * Math.cos(rotation.x);
    cameraRef.current.position.y = radius * Math.sin(-rotation.x) + 5;
    cameraRef.current.position.z = radius * Math.cos(rotation.y) * Math.cos(rotation.x);
    cameraRef.current.lookAt(0, explodeLevel * 0.5, 0);
  }, [rotation, zoom, explodeLevel]);

  // Mouse interaction with raycasting
  const handleMouseDown = (e) => {
    setIsDragging(true);
    setLastMouse({ x: e.clientX, y: e.clientY });
  };

  const handleMouseMove = (e) => {
    // Update mouse position for raycasting
    const rect = containerRef.current?.getBoundingClientRect();
    if (rect) {
      mouseRef.current.x = ((e.clientX - rect.left) / rect.width) * 2 - 1;
      mouseRef.current.y = -((e.clientY - rect.top) / rect.height) * 2 + 1;

      // Raycast for hover effect
      if (!isDragging && sceneRef.current && cameraRef.current) {
        raycasterRef.current.setFromCamera(mouseRef.current, cameraRef.current);
        const allObjects = [];
        Object.values(layersRef.current).forEach(group => {
          group.children.forEach(child => allObjects.push(child));
        });
        
        const intersects = raycasterRef.current.intersectObjects(allObjects);
        if (intersects.length > 0 && intersects[0].object.userData.component) {
          setHoveredComponent(intersects[0].object.userData.component);
          // Track tile ID if hovering over a tile component
          const tileId = intersects[0].object.userData.tileId;
          if (tileId !== undefined) {
            setActiveTile(tileId);
          } else {
            setActiveTile(null);
          }
        } else {
          setHoveredComponent(null);
          setActiveTile(null);
        }
      }
    }

    if (!isDragging) return;
    const deltaX = e.clientX - lastMouse.x;
    const deltaY = e.clientY - lastMouse.y;
    setRotation(prev => ({
      x: Math.max(-Math.PI/2 + 0.1, Math.min(0.1, prev.x + deltaY * 0.005)),
      y: prev.y + deltaX * 0.005
    }));
    setLastMouse({ x: e.clientX, y: e.clientY });
  };

  const handleMouseUp = () => setIsDragging(false);
  
  const handleWheel = (e) => {
    e.preventDefault();
    setZoom(prev => Math.max(0.5, Math.min(3, prev - e.deltaY * 0.001)));
  };

  const handleClick = (e) => {
    if (Math.abs(e.clientX - lastMouse.x) > 5 || Math.abs(e.clientY - lastMouse.y) > 5) return;
    
    const rect = containerRef.current?.getBoundingClientRect();
    if (!rect) return;
    
    mouseRef.current.x = ((e.clientX - rect.left) / rect.width) * 2 - 1;
    mouseRef.current.y = -((e.clientY - rect.top) / rect.height) * 2 + 1;

    raycasterRef.current.setFromCamera(mouseRef.current, cameraRef.current);
    const allObjects = [];
    Object.values(layersRef.current).forEach(group => {
      group.children.forEach(child => allObjects.push(child));
    });
    
    const intersects = raycasterRef.current.intersectObjects(allObjects);
    if (intersects.length > 0 && intersects[0].object.userData.component) {
      setSelectedComponent(intersects[0].object.userData.component);
    }
  };

  // Component creators (enhanced versions)
  function createSubstrate(scene) {
    const group = new THREE.Group();
    
    const packageGeo = new THREE.BoxGeometry(10, 0.3, 10);
    const packageMat = new THREE.MeshStandardMaterial({ 
      color: 0x1a472a, 
      roughness: 0.7, 
      metalness: 0.2,
      envMapIntensity: 0.5
    });
    const packageMesh = new THREE.Mesh(packageGeo, packageMat);
    packageMesh.position.y = -0.5;
    packageMesh.receiveShadow = true;
    packageMesh.castShadow = true;
    packageMesh.userData.component = 'substrate';
    group.add(packageMesh);

    // Add circuit traces
    for (let i = 0; i < 20; i++) {
      const traceGeo = new THREE.BoxGeometry(9, 0.01, 0.02);
      const traceMat = new THREE.MeshStandardMaterial({ 
        color: 0x8b7355, 
        roughness: 0.3, 
        metalness: 0.8,
        emissive: 0x4a3020,
        emissiveIntensity: 0.1
      });
      const trace = new THREE.Mesh(traceGeo, traceMat);
      trace.position.set(0, -0.33, -4.5 + i * 0.5);
      group.add(trace);
    }

    const frameGeo = new THREE.BoxGeometry(10.4, 0.35, 10.4);
    const frameMat = new THREE.MeshStandardMaterial({ 
      color: 0x0d2818, 
      roughness: 0.5, 
      metalness: 0.3 
    });
    const frameMesh = new THREE.Mesh(frameGeo, frameMat);
    frameMesh.position.y = -0.52;
    frameMesh.userData.component = 'substrate';
    group.add(frameMesh);

    scene.add(group);
    return group;
  }

  function createBGA(scene) {
    const group = new THREE.Group();
    const ballMat = new THREE.MeshStandardMaterial({ 
      color: 0xd0d0d0, 
      roughness: 0.15, 
      metalness: 0.95,
      envMapIntensity: 1.0
    });
    const ballGeo = new THREE.SphereGeometry(0.12, 20, 20);

    const gridSize = 12;
    const spacing = 0.7;
    const startPos = -(gridSize - 1) * spacing / 2;

    for (let row = 0; row < gridSize; row++) {
      for (let col = 0; col < gridSize; col++) {
        const x = startPos + col * spacing;
        const z = startPos + row * spacing;
        if (Math.abs(x) < 2 && Math.abs(z) < 2) continue;

        const ball = new THREE.Mesh(ballGeo, ballMat);
        ball.position.set(x, -0.75, z);
        ball.castShadow = true;
        ball.userData.component = 'bga';
        group.add(ball);
      }
    }

    scene.add(group);
    return group;
  }

  function createBondWires(scene) {
    const group = new THREE.Group();
    const wireMat = new THREE.MeshStandardMaterial({ 
      color: 0xffd700, 
      roughness: 0.1, 
      metalness: 0.95,
      emissive: 0xffaa00,
      emissiveIntensity: 0.05
    });

    const dieEdge = 3;
    const packageEdge = 4.5;
    const wireCount = 25;

    for (let side = 0; side < 4; side++) {
      for (let i = 0; i < wireCount; i++) {
        const t = (i / (wireCount - 1)) * 2 - 1;
        
        let startX, startZ, endX, endZ;
        if (side === 0) { startX = t * dieEdge; startZ = -dieEdge; endX = t * packageEdge; endZ = -packageEdge; }
        else if (side === 1) { startX = dieEdge; startZ = t * dieEdge; endX = packageEdge; endZ = t * packageEdge; }
        else if (side === 2) { startX = t * dieEdge; startZ = dieEdge; endX = t * packageEdge; endZ = packageEdge; }
        else { startX = -dieEdge; startZ = t * dieEdge; endX = -packageEdge; endZ = t * packageEdge; }

        const curve = new THREE.QuadraticBezierCurve3(
          new THREE.Vector3(startX, -0.1, startZ),
          new THREE.Vector3((startX + endX) / 2, 0.4, (startZ + endZ) / 2),
          new THREE.Vector3(endX, -0.35, endZ)
        );
        
        const tubeGeo = new THREE.TubeGeometry(curve, 16, 0.01, 8, false);
        const wireMesh = new THREE.Mesh(tubeGeo, wireMat);
        wireMesh.castShadow = true;
        wireMesh.userData.component = 'bondwires';
        group.add(wireMesh);
      }
    }

    scene.add(group);
    return group;
  }

  function createDie(scene) {
    const group = new THREE.Group();
    
    const dieGeo = new THREE.BoxGeometry(6, 0.15, 6);
    const dieMat = new THREE.MeshStandardMaterial({ 
      color: 0x2a2a3a, 
      roughness: 0.2, 
      metalness: 0.8,
      envMapIntensity: 0.8
    });
    const dieMesh = new THREE.Mesh(dieGeo, dieMat);
    dieMesh.position.y = -0.25;
    dieMesh.castShadow = true;
    dieMesh.receiveShadow = true;
    dieMesh.userData.component = 'die';
    group.add(dieMesh);

    // Detailed surface textures
    const traceGeo = new THREE.PlaneGeometry(5.8, 5.8, 32, 32);
    const traceMat = new THREE.MeshStandardMaterial({ 
      color: 0x3d3d50, 
      roughness: 0.3, 
      metalness: 0.7, 
      transparent: true, 
      opacity: 0.9
    });
    const traceMesh = new THREE.Mesh(traceGeo, traceMat);
    traceMesh.rotation.x = -Math.PI / 2;
    traceMesh.position.y = -0.16;
    traceMesh.userData.component = 'die';
    group.add(traceMesh);

    scene.add(group);
    return group;
  }

  function createTileArray(scene) {
    const group = new THREE.Group();
    const tileSize = 0.6;
    const gap = 0.08;
    const startX = -(TILE_DIM - 1) * (tileSize + gap) / 2;
    const startZ = -(TILE_DIM - 1) * (tileSize + gap) / 2;

    // Create all 64 tiles with proper T00-T63 labeling - NO CENTER GAP (real floorplan)
    for (let row = 0; row < TILE_DIM; row++) {
      for (let col = 0; col < TILE_DIM; col++) {
        const tileId = row * TILE_DIM + col;
        const tileLabel = `T${tileId.toString().padStart(2, '0')}`;
        const x = startX + col * (tileSize + gap);
        const z = startZ + row * (tileSize + gap);

        // Determine activation wave for thermal coloring
        let waveIndex = -1;
        ACTIVATION_WAVES.forEach((wave, idx) => {
          if (wave.includes(tileId)) waveIndex = idx;
        });

        // Enhanced tile base (rpu_tile_shell)
        const tileGeo = new THREE.BoxGeometry(tileSize, 0.08, tileSize);
        const tileMat = new THREE.MeshStandardMaterial({ 
          color: 0x1a1f2e, 
          roughness: 0.4, 
          metalness: 0.5,
          emissive: 0x000000,
          emissiveIntensity: 0
        });
        const tileMesh = new THREE.Mesh(tileGeo, tileMat);
        tileMesh.position.set(x, 0, z);
        tileMesh.castShadow = true;
        tileMesh.receiveShadow = true;
        tileMesh.userData = {
          component: 'tile',
          tileId: tileId,
          tileLabel: tileLabel,
          waveIndex: waveIndex,
          originalTransparent: false,
          originalOpacity: 1,
          originalColor: 0x1a1f2e
        };
        group.add(tileMesh);

        // phi_rft_core (main compute unit)
        const coreGeo = new THREE.BoxGeometry(tileSize * 0.45, 0.06, tileSize * 0.4);
        const coreMat = new THREE.MeshStandardMaterial({ 
          color: CLOCK_DOMAINS.clk_tile.color, 
          roughness: 0.2, 
          metalness: 0.7, 
          emissive: CLOCK_DOMAINS.clk_tile.color, 
          emissiveIntensity: 0.2 
        });
        const coreMesh = new THREE.Mesh(coreGeo, coreMat);
        coreMesh.position.set(x - tileSize * 0.15, 0.07, z - tileSize * 0.15);
        coreMesh.castShadow = true;
        coreMesh.userData = {
          component: 'phirft',
          tileId: tileId,
          clockDomain: 'clk_tile',
          originalTransparent: false,
          originalOpacity: 1
        };
        group.add(coreMesh);

        // Φ symbol indicator on core
        const phiGeo = new THREE.PlaneGeometry(tileSize * 0.2, tileSize * 0.2);
        const phiMat = new THREE.MeshStandardMaterial({ 
          color: 0xffffff, 
          transparent: true, 
          opacity: 0.7,
          emissive: 0x00ffcc,
          emissiveIntensity: 0.4
        });
        const phiMesh = new THREE.Mesh(phiGeo, phiMat);
        phiMesh.rotation.x = -Math.PI / 2;
        phiMesh.position.set(x - tileSize * 0.15, 0.11, z - tileSize * 0.15);
        group.add(phiMesh);

        // scratch[64] SRAM
        const scrGeo = new THREE.BoxGeometry(tileSize * 0.22, 0.05, tileSize * 0.16);
        const scrMat = new THREE.MeshStandardMaterial({ 
          color: CLOCK_DOMAINS.clk_tile.color, 
          roughness: 0.3, 
          metalness: 0.6, 
          emissive: 0x4a9eff, 
          emissiveIntensity: 0.15 
        });
        const scrMesh = new THREE.Mesh(scrGeo, scrMat);
        scrMesh.position.set(x + tileSize * 0.22, 0.06, z - tileSize * 0.22);
        scrMesh.castShadow = true;
        scrMesh.userData = {
          component: 'scratchpad',
          tileId: tileId,
          originalTransparent: false,
          originalOpacity: 1
        };
        group.add(scrMesh);

        // topo_mem[64] H³ vertex memory
        const topoGeo = new THREE.BoxGeometry(tileSize * 0.22, 0.05, tileSize * 0.16);
        const topoMat = new THREE.MeshStandardMaterial({ 
          color: 0xff6b9d, 
          roughness: 0.3, 
          metalness: 0.6, 
          emissive: 0xff6b9d, 
          emissiveIntensity: 0.15 
        });
        const topoMesh = new THREE.Mesh(topoGeo, topoMat);
        topoMesh.position.set(x + tileSize * 0.22, 0.06, z + tileSize * 0.02);
        topoMesh.castShadow = true;
        topoMesh.userData = {
          component: 'topo',
          tileId: tileId,
          originalTransparent: false,
          originalOpacity: 1
        };
        group.add(topoMesh);

        // Wormhole NoC Router (clk_noc domain)
        const routerGeo = new THREE.BoxGeometry(tileSize * 0.75, 0.04, tileSize * 0.25);
        const routerMat = new THREE.MeshStandardMaterial({ 
          color: CLOCK_DOMAINS.clk_noc.color, 
          roughness: 0.3, 
          metalness: 0.6, 
          emissive: CLOCK_DOMAINS.clk_noc.color, 
          emissiveIntensity: 0.1 
        });
        const routerMesh = new THREE.Mesh(routerGeo, routerMat);
        routerMesh.position.set(x, 0.065, z + tileSize * 0.28);
        routerMesh.castShadow = true;
        routerMesh.userData = {
          component: 'router',
          tileId: tileId,
          clockDomain: 'clk_noc',
          originalTransparent: false,
          originalOpacity: 1
        };
        group.add(routerMesh);

        // 5-port router indicators (N/S/E/W/Local)
        const portPositions = [
          { x: 0, z: -0.08, label: 'N' },
          { x: 0, z: 0.08, label: 'S' },
          { x: 0.15, z: 0, label: 'E' },
          { x: -0.15, z: 0, label: 'W' },
        ];
        portPositions.forEach(port => {
          const portGeo = new THREE.BoxGeometry(0.03, 0.02, 0.03);
          const portMat = new THREE.MeshStandardMaterial({
            color: 0x2ea043,
            emissive: 0x2ea043,
            emissiveIntensity: 0.3
          });
          const portMesh = new THREE.Mesh(portGeo, portMat);
          portMesh.position.set(x + port.x * tileSize, 0.085, z + tileSize * 0.28 + port.z);
          group.add(portMesh);
        });
      }
    }

    scene.add(group);
    return group;
  }

  function createNoC(scene) {
    const group = new THREE.Group();
    const tileSize = 0.6;
    const gap = 0.08;
    const startX = -(TILE_DIM - 1) * (tileSize + gap) / 2;
    const startZ = -(TILE_DIM - 1) * (tileSize + gap) / 2;
    const nocY = 0.13;

    const nocMat = new THREE.MeshStandardMaterial({ 
      color: 0x4a9eff, 
      roughness: 0.2, 
      metalness: 0.85, 
      emissive: 0x4a9eff, 
      emissiveIntensity: 0.25 
    });

    // Horizontal connections
    for (let row = 0; row < TILE_DIM; row++) {
      for (let col = 0; col < TILE_DIM - 1; col++) {
        const x1 = startX + col * (tileSize + gap) + tileSize / 2;
        const x2 = startX + (col + 1) * (tileSize + gap) - tileSize / 2;
        const z = startZ + row * (tileSize + gap);
        
        const length = x2 - x1;
        const lineGeo = new THREE.BoxGeometry(length, 0.02, 0.04);
        const lineMesh = new THREE.Mesh(lineGeo, nocMat);
        lineMesh.position.set((x1 + x2) / 2, nocY, z + tileSize * 0.25);
        lineMesh.castShadow = true;
        lineMesh.userData.component = 'noc';
        lineMesh.userData.originalTransparent = false;
        lineMesh.userData.originalOpacity = 1;
        group.add(lineMesh);
      }
    }

    // Vertical connections
    for (let row = 0; row < TILE_DIM - 1; row++) {
      for (let col = 0; col < TILE_DIM; col++) {
        const x = startX + col * (tileSize + gap);
        const z1 = startZ + row * (tileSize + gap) + tileSize / 2;
        const z2 = startZ + (row + 1) * (tileSize + gap) - tileSize / 2;
        
        const length = z2 - z1;
        const lineGeo = new THREE.BoxGeometry(0.04, 0.02, length);
        const lineMesh = new THREE.Mesh(lineGeo, nocMat);
        lineMesh.position.set(x, nocY, (z1 + z2) / 2);
        lineMesh.castShadow = true;
        lineMesh.userData.component = 'noc';
        lineMesh.userData.originalTransparent = false;
        lineMesh.userData.originalOpacity = 1;
        group.add(lineMesh);
      }
    }

    // Enhanced animated particles
    const particles = [];
    const particleMat = new THREE.MeshStandardMaterial({ 
      color: 0xffffff, 
      emissive: 0x00d4aa, 
      emissiveIntensity: 1.0,
      transparent: true,
      opacity: 0.9
    });
    const particleGeo = new THREE.SphereGeometry(0.03, 12, 12);

    for (let i = 0; i < 40; i++) {
      const particle = new THREE.Mesh(particleGeo, particleMat.clone());
      const row = Math.floor(Math.random() * TILE_DIM);
      const col = Math.floor(Math.random() * (TILE_DIM - 1));
      const horizontal = Math.random() > 0.5;
      
      if (horizontal) {
        const x1 = startX + col * (tileSize + gap);
        const x2 = startX + (col + 1) * (tileSize + gap);
        const z = startZ + row * (tileSize + gap) + tileSize * 0.25;
        particle.userData.path = {
          start: new THREE.Vector3(x1, nocY + 0.03, z),
          end: new THREE.Vector3(x2, nocY + 0.03, z)
        };
      } else {
        const x = startX + col * (tileSize + gap);
        const z1 = startZ + row * (tileSize + gap);
        const z2 = startZ + (row + 1) * (tileSize + gap);
        particle.userData.path = {
          start: new THREE.Vector3(x, nocY + 0.03, z1),
          end: new THREE.Vector3(x, nocY + 0.03, z2)
        };
      }
      
      particle.position.copy(particle.userData.path.start);
      group.add(particle);
      particles.push(particle);
    }
    group.userData.particles = particles;

    scene.add(group);
    return group;
  }

  function createHeatspreader(scene) {
    const group = new THREE.Group();
    
    const ihsGeo = new THREE.BoxGeometry(5.5, 0.18, 5.5);
    const ihsMat = new THREE.MeshStandardMaterial({ 
      color: 0x9a9a9a, 
      roughness: 0.25, 
      metalness: 0.85,
      envMapIntensity: 1.2
    });
    const ihsMesh = new THREE.Mesh(ihsGeo, ihsMat);
    ihsMesh.position.y = 0.26;
    ihsMesh.castShadow = true;
    ihsMesh.receiveShadow = true;
    ihsMesh.userData.component = 'heatspreader';
    ihsMesh.userData.originalTransparent = false;
    ihsMesh.userData.originalOpacity = 1;
    group.add(ihsMesh);

    // Brushed metal texture simulation
    for (let i = 0; i < 30; i++) {
      const lineGeo = new THREE.BoxGeometry(5.4, 0.005, 0.01);
      const lineMat = new THREE.MeshStandardMaterial({ 
        color: 0xaaaaaa, 
        roughness: 0.2, 
        metalness: 0.9,
        transparent: true,
        opacity: 0.3
      });
      const line = new THREE.Mesh(lineGeo, lineMat);
      line.position.set(0, 0.36, -2.7 + i * 0.18);
      group.add(line);
    }

    // Beveled edge
    const bevelGeo = new THREE.BoxGeometry(5.7, 0.1, 5.7);
    const bevelMat = new THREE.MeshStandardMaterial({ 
      color: 0x7a7a7a, 
      roughness: 0.35, 
      metalness: 0.75 
    });
    const bevelMesh = new THREE.Mesh(bevelGeo, bevelMat);
    bevelMesh.position.y = 0.16;
    bevelMesh.castShadow = true;
    bevelMesh.userData.component = 'heatspreader';
    group.add(bevelMesh);

    scene.add(group);
    return group;
  }

  // Create SPINE modules on right edge of die (per PHYSICAL_DESIGN_SPEC floorplan)
  function createSpineModules(scene) {
    const group = new THREE.Group();
    const spineX = 3.2; // Right edge of tile array
    
    // quantoniumos_unified_core (Top of spine) - 420K gates
    const unifiedGeo = new THREE.BoxGeometry(0.8, 0.12, 1.0);
    const unifiedMat = new THREE.MeshStandardMaterial({ 
      color: 0x9b59b6, 
      roughness: 0.25, 
      metalness: 0.7,
      emissive: 0x9b59b6, 
      emissiveIntensity: 0.3
    });
    const unifiedMesh = new THREE.Mesh(unifiedGeo, unifiedMat);
    unifiedMesh.position.set(spineX, 0.09, -1.8);
    unifiedMesh.castShadow = true;
    unifiedMesh.userData = { 
      component: 'unified_controller',
      originalTransparent: false,
      originalOpacity: 1
    };
    group.add(unifiedMesh);
    
    // rft_sis_hash_v31 (Center of spine) - 320K gates, clk_sis domain
    const sisGeo = new THREE.BoxGeometry(0.8, 0.14, 2.0);
    const sisMat = new THREE.MeshStandardMaterial({ 
      color: CLOCK_DOMAINS.clk_sis.color, 
      roughness: 0.25, 
      metalness: 0.7,
      emissive: CLOCK_DOMAINS.clk_sis.color, 
      emissiveIntensity: 0.25
    });
    const sisMesh = new THREE.Mesh(sisGeo, sisMat);
    sisMesh.position.set(spineX, 0.10, 0);
    sisMesh.castShadow = true;
    sisMesh.userData = { 
      component: 'sis_hash',
      clockDomain: 'clk_sis',
      originalTransparent: false,
      originalOpacity: 1
    };
    group.add(sisMesh);
    
    // feistel_48_cipher (Below SIS) - 45K gates, clk_feistel domain
    const feistelGeo = new THREE.BoxGeometry(0.8, 0.10, 1.5);
    const feistelMat = new THREE.MeshStandardMaterial({ 
      color: CLOCK_DOMAINS.clk_feistel.color, 
      roughness: 0.2, 
      metalness: 0.75,
      emissive: CLOCK_DOMAINS.clk_feistel.color, 
      emissiveIntensity: 0.35
    });
    const feistelMesh = new THREE.Mesh(feistelGeo, feistelMat);
    feistelMesh.position.set(spineX, 0.08, 1.75);
    feistelMesh.castShadow = true;
    feistelMesh.userData = { 
      component: 'feistel',
      clockDomain: 'clk_feistel',
      originalTransparent: false,
      originalOpacity: 1
    };
    group.add(feistelMesh);

    // SPINE label
    const spineGeo = new THREE.BoxGeometry(0.15, 0.02, 4.5);
    const spineMat = new THREE.MeshStandardMaterial({ 
      color: 0x2c3e50, 
      roughness: 0.5, 
      metalness: 0.4
    });
    const spineMesh = new THREE.Mesh(spineGeo, spineMat);
    spineMesh.position.set(spineX + 0.5, 0.04, 0);
    group.add(spineMesh);

    scene.add(group);
    return group;
  }

  // Create DMA Ingress on south edge (per floorplan)
  function createDMAIngress(scene) {
    const group = new THREE.Group();
    const dmaZ = 3.0; // South edge
    
    // rpu_dma_ingress - 6.5K gates, wide aspect ratio
    const dmaGeo = new THREE.BoxGeometry(5.5, 0.06, 0.5);
    const dmaMat = new THREE.MeshStandardMaterial({ 
      color: 0xe74c3c, 
      roughness: 0.3, 
      metalness: 0.6,
      emissive: 0xe74c3c, 
      emissiveIntensity: 0.15
    });
    const dmaMesh = new THREE.Mesh(dmaGeo, dmaMat);
    dmaMesh.position.set(0, 0.05, dmaZ);
    dmaMesh.castShadow = true;
    dmaMesh.userData = { 
      component: 'dma_ingress',
      originalTransparent: false,
      originalOpacity: 1
    };
    group.add(dmaMesh);

    scene.add(group);
    return group;
  }

  // Create PLL islands (NW and SE corners per floorplan)
  function createPLLIslands(scene) {
    const group = new THREE.Group();
    
    // PLL_NW (northwest corner)
    const pllGeo = new THREE.CylinderGeometry(0.25, 0.25, 0.08, 32);
    const pllMat = new THREE.MeshStandardMaterial({ 
      color: 0x27ae60, 
      roughness: 0.25, 
      metalness: 0.7,
      emissive: 0x27ae60, 
      emissiveIntensity: 0.25
    });
    
    const pllNW = new THREE.Mesh(pllGeo, pllMat);
    pllNW.position.set(-2.8, 0.06, -2.8);
    pllNW.userData = { 
      component: 'pll',
      originalTransparent: false,
      originalOpacity: 1
    };
    group.add(pllNW);
    
    // PLL_SE (southeast corner)
    const pllSE = new THREE.Mesh(pllGeo, pllMat.clone());
    pllSE.position.set(2.8, 0.06, 2.8);
    pllSE.userData = { 
      component: 'pll',
      originalTransparent: false,
      originalOpacity: 1
    };
    group.add(pllSE);

    scene.add(group);
    return group;
  }

  function createQLogo(scene) {
    const group = new THREE.Group();
    
    // Q outer ring with segments - positioned at center as branding overlay
    const ringGeo = new THREE.TorusGeometry(0.6, 0.12, 20, 48);
    const ringMat = new THREE.MeshStandardMaterial({ 
      color: 0x00d4aa, 
      roughness: 0.15, 
      metalness: 0.85,
      emissive: 0x00d4aa, 
      emissiveIntensity: 0.4,
      transparent: true,
      opacity: 0.8
    });
    const ringMesh = new THREE.Mesh(ringGeo, ringMat);
    ringMesh.rotation.x = -Math.PI / 2;
    ringMesh.position.y = 0.22;
    ringMesh.castShadow = true;
    ringMesh.userData.component = 'qlogo';
    ringMesh.userData.originalTransparent = true;
    ringMesh.userData.originalOpacity = 0.8;
    group.add(ringMesh);

    // Q tail
    const tailGeo = new THREE.BoxGeometry(0.12, 0.12, 0.5);
    const tailMat = new THREE.MeshStandardMaterial({ 
      color: 0x00d4aa, 
      roughness: 0.15, 
      metalness: 0.85,
      emissive: 0x00d4aa, 
      emissiveIntensity: 0.4,
      transparent: true,
      opacity: 0.8
    });
    const tailMesh = new THREE.Mesh(tailGeo, tailMat);
    tailMesh.position.set(0.4, 0.22, 0.4);
    tailMesh.rotation.y = Math.PI / 4;
    tailMesh.castShadow = true;
    tailMesh.userData.component = 'qlogo';
    group.add(tailMesh);

    // Enhanced glow disc with gradient
    const glowGeo = new THREE.CircleGeometry(0.45, 48);
    const glowMat = new THREE.MeshStandardMaterial({ 
      color: 0x00ffdd, 
      transparent: true, 
      opacity: 0.3,
      emissive: 0x00ffcc, 
      emissiveIntensity: 0.5, 
      side: THREE.DoubleSide
    });
    const glowMesh = new THREE.Mesh(glowGeo, glowMat);
    glowMesh.rotation.x = -Math.PI / 2;
    glowMesh.position.y = 0.18;
    glowMesh.userData.originalTransparent = true;
    glowMesh.userData.originalOpacity = 0.3;
    group.add(glowMesh);
    group.userData.glowMesh = glowMesh;

    scene.add(group);
    return group;
  }

  const toggleLayer = (layer) => {
    setShowLayers(prev => ({ ...prev, [layer]: !prev[layer] }));
  };

  const startTour = () => {
    setTourActive(true);
    setTourStep(0);
  };

  const stopTour = () => {
    setTourActive(false);
    setTourStep(0);
  };

  const resetView = () => {
    setTargetRotation({ x: -0.35, y: 0.4 });
    setTargetZoom(1);
    setTargetExplode(0);
  };

  // ===============================================
  // BENCHMARK RUNNER FUNCTIONS
  // ===============================================
  const runBenchmark = useCallback((scenarioId) => {
    if (benchmarkRunning) return;
    
    setBenchmarkMode(scenarioId);
    setBenchmarkRunning(true);
    setBenchmarkProgress(0);
    setShowThermalView(true);
    
    const scenario = Object.values(BENCHMARK_SCENARIOS).find(s => s.id === scenarioId);
    const duration = scenarioId === 'rft' ? 3000 : scenarioId === 'h3' ? 4000 : scenarioId === 'sis' ? 2500 : 2000;
    
    let progress = 0;
    const tilesPerWave = Math.ceil(scenario?.tiles || 64) / 9;
    
    benchmarkIntervalRef.current = setInterval(() => {
      progress += 2;
      setBenchmarkProgress(progress);
      
      // Update live metrics during benchmark
      const utilization = Math.min(progress * 1.5, 94.2);
      setLiveMetrics({
        currentTOPS: (2.39 * progress / 100).toFixed(2),
        currentPower: (0.54 + (8.2 - 0.54) * progress / 100).toFixed(1),
        activeTiles: Math.floor((progress / 100) * (scenario?.tiles || 64)),
        nocUtilization: Math.min(progress * 1.2, 85)
      });
      
      // Animate thermal waves
      setThermalWaveIndex(Math.floor((progress / 100) * 9));
      
      // RFT pipeline stages
      if (scenarioId === 'rft') {
        setRftStage(Math.floor((progress / 100) * 5));
      }
      
      if (progress >= 100) {
        clearInterval(benchmarkIntervalRef.current);
        setBenchmarkRunning(false);
        setLiveMetrics({
          currentTOPS: BENCHMARK_METRICS.peakPerformance.value,
          currentPower: BENCHMARK_METRICS.powerDissipation.value,
          activeTiles: scenario?.tiles || 64,
          nocUtilization: 85
        });
      }
    }, duration / 50);
  }, [benchmarkRunning]);

  const stopBenchmark = useCallback(() => {
    if (benchmarkIntervalRef.current) {
      clearInterval(benchmarkIntervalRef.current);
    }
    setBenchmarkRunning(false);
    setBenchmarkMode(null);
    setBenchmarkProgress(0);
    setThermalWaveIndex(0);
    setRftStage(0);
    setLiveMetrics({
      currentTOPS: 0,
      currentPower: 0.54,
      activeTiles: 0,
      nocUtilization: 0
    });
  }, []);

  // Thermal wave animation effect
  useEffect(() => {
    if (showThermalView && !benchmarkRunning) {
      thermalIntervalRef.current = setInterval(() => {
        setThermalWaveIndex(prev => (prev + 1) % 9);
      }, 200);
      return () => clearInterval(thermalIntervalRef.current);
    }
  }, [showThermalView, benchmarkRunning]);

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      if (benchmarkIntervalRef.current) clearInterval(benchmarkIntervalRef.current);
      if (thermalIntervalRef.current) clearInterval(thermalIntervalRef.current);
    };
  }, []);

  // Dismiss onboarding
  const dismissOnboarding = useCallback(() => {
    setShowOnboarding(false);
    localStorage.setItem('rftpu-onboarding-seen', 'true');
  }, []);
  
  // Check if onboarding was previously dismissed
  useEffect(() => {
    if (localStorage.getItem('rftpu-onboarding-seen')) {
      setShowOnboarding(false);
    }
  }, []);

  return (
    <div className="w-full h-screen bg-gradient-to-br from-gray-950 via-gray-900 to-gray-950 relative overflow-hidden flex">
      {/* Loading Screen */}
      {isLoading && (
        <div className="absolute inset-0 z-50 bg-gray-950 flex flex-col items-center justify-center">
          <div className="relative">
            <div className="w-24 h-24 border-4 border-cyan-500/20 rounded-full animate-ping absolute" />
            <div className="w-24 h-24 border-4 border-t-cyan-400 border-r-cyan-400 border-b-transparent border-l-transparent rounded-full animate-spin" />
            <span className="absolute inset-0 flex items-center justify-center text-4xl text-cyan-300 font-bold">Φ</span>
          </div>
          <p className="mt-8 text-cyan-400 font-mono text-lg animate-pulse">Initializing RFT-Processor...</p>
          <div className="mt-4 flex items-center gap-2 text-xs text-gray-500 font-mono">
            <span className="w-2 h-2 bg-cyan-500 rounded-full animate-pulse" />
            Loading 64 tiles • NoC fabric • SPINE modules
          </div>
        </div>
      )}
      
      {/* Onboarding Overlay */}
      {showOnboarding && !isLoading && (
        <div className="absolute inset-0 z-40 bg-gray-950/90 backdrop-blur-xl flex items-center justify-center p-8">
          <div className="max-w-2xl bg-gray-900/95 rounded-2xl border border-cyan-500/30 p-8 shadow-2xl shadow-cyan-500/10 animate-in fade-in slide-in-from-bottom-4 duration-500">
            <div className="flex items-center gap-4 mb-6">
              <span className="text-5xl text-cyan-300 drop-shadow-[0_0_15px_rgba(34,211,238,0.5)]">Φ</span>
              <div>
                <h2 className="text-2xl font-bold text-white font-mono">RFT-PROCESSOR</h2>
                <p className="text-cyan-400/70 text-sm">Interactive 3D Architecture Visualization</p>
              </div>
            </div>
            
            <div className="grid grid-cols-3 gap-4 mb-8">
              <div className="bg-gray-800/50 rounded-xl p-4 border border-gray-700">
                <div className="text-3xl mb-2">🖱️</div>
                <h3 className="text-sm font-bold text-white mb-1">Orbit & Pan</h3>
                <p className="text-xs text-gray-400">Drag to rotate, scroll to zoom, right-click to pan</p>
              </div>
              <div className="bg-gray-800/50 rounded-xl p-4 border border-gray-700">
                <div className="text-3xl mb-2">👆</div>
                <h3 className="text-sm font-bold text-white mb-1">Click Components</h3>
                <p className="text-xs text-gray-400">Select tiles, routers, and modules for detailed specs</p>
              </div>
              <div className="bg-gray-800/50 rounded-xl p-4 border border-gray-700">
                <div className="text-3xl mb-2">🎬</div>
                <h3 className="text-sm font-bold text-white mb-1">Guided Tour</h3>
                <p className="text-xs text-gray-400">Automatic walkthrough of all architecture layers</p>
              </div>
            </div>
            
            <div className="flex items-center justify-between">
              <div className="text-xs text-gray-500 font-mono">
                USPTO #19/169,399 • DOI: {PAPER_REFERENCES.doi}
              </div>
              <button
                onClick={dismissOnboarding}
                className="px-6 py-3 bg-gradient-to-r from-cyan-600 to-blue-600 text-white font-bold rounded-xl hover:from-cyan-500 hover:to-blue-500 transition-all shadow-lg shadow-cyan-500/30 flex items-center gap-2"
              >
                <span>Explore Architecture</span>
                <span>→</span>
              </button>
            </div>
          </div>
        </div>
      )}
      {/* 3D Canvas */}
      <div 
        ref={containerRef}
        className="flex-1 cursor-grab active:cursor-grabbing"
        onMouseDown={handleMouseDown}
        onMouseMove={handleMouseMove}
        onMouseUp={handleMouseUp}
        onMouseLeave={handleMouseUp}
        onClick={handleClick}
        onWheel={handleWheel}
      />

      {/* Enhanced Header with DOI */}
      <div className="absolute top-6 left-6 z-10">
        <div className="bg-gray-900/90 backdrop-blur-xl rounded-xl p-4 border border-cyan-500/30 shadow-2xl shadow-cyan-500/20">
          <h1 className="text-3xl font-bold text-transparent bg-clip-text bg-gradient-to-r from-cyan-400 to-cyan-200 font-mono tracking-wide flex items-center gap-3">
            <span className="text-4xl text-cyan-300 drop-shadow-[0_0_10px_rgba(34,211,238,0.5)]">Φ</span>
            RFT-PROCESSOR
          </h1>
          <p className="text-sm text-gray-300 font-mono mt-2 flex items-center gap-2">
            <span className="w-2 h-2 bg-green-400 rounded-full animate-pulse"></span>
            Resonant Fourier Transform Processor • 64 Tiles
          </p>
          <p className="text-xs text-gray-400 font-mono mt-1">
            USPTO Patent #19/169,399 • TSMC N7FF (7nm FinFET)
          </p>
          <div className="flex items-center gap-2 mt-1">
            <span className="text-xs text-cyan-400/70 font-mono">© 2025 Luis M. Minier</span>
            <a href={`https://doi.org/${PAPER_REFERENCES.doi}`} target="_blank" rel="noopener noreferrer" 
               className="text-xs bg-blue-600/80 px-2 py-0.5 rounded font-mono hover:bg-blue-500 transition-colors">
              DOI: {PAPER_REFERENCES.doi}
            </a>
          </div>
        </div>
      </div>

      {/* Live Performance Counter */}
      {(benchmarkRunning || showBenchmarkPanel) && (
        <div className="absolute top-6 left-1/2 transform -translate-x-1/2 z-20">
          <div className="bg-gray-900/95 backdrop-blur-xl rounded-xl p-4 border border-purple-500/50 shadow-2xl shadow-purple-500/20">
            <div className="text-center mb-3">
              <div className="text-3xl font-bold font-mono text-transparent bg-clip-text bg-gradient-to-r from-purple-400 to-pink-400">
                {liveMetrics.currentTOPS} <span className="text-lg">TOPS</span>
              </div>
              <div className="text-xs text-gray-400 font-mono">Real-time Performance</div>
            </div>
            {benchmarkRunning && (
              <div className="space-y-2">
                <div className="flex justify-between text-xs font-mono">
                  <span className="text-gray-400">Progress:</span>
                  <span className="text-purple-300">{benchmarkProgress}%</span>
                </div>
                <div className="w-48 h-2 bg-gray-700 rounded-full overflow-hidden">
                  <div 
                    className="h-full bg-gradient-to-r from-purple-500 to-pink-500 transition-all duration-100"
                    style={{ width: `${benchmarkProgress}%` }}
                  />
                </div>
                <div className="grid grid-cols-2 gap-2 text-xs font-mono mt-2">
                  <div className="text-center p-1 bg-gray-800/50 rounded">
                    <div className="text-cyan-300">{liveMetrics.activeTiles}</div>
                    <div className="text-gray-500 text-[10px]">Active Tiles</div>
                  </div>
                  <div className="text-center p-1 bg-gray-800/50 rounded">
                    <div className="text-orange-300">{liveMetrics.currentPower}W</div>
                    <div className="text-gray-500 text-[10px]">Power</div>
                  </div>
                </div>
              </div>
            )}
          </div>
        </div>
      )}

      {/* Performance Metrics Panel */}
      {showMetrics && (
        <div className="absolute top-6 right-[21rem] z-10">
          <div className="bg-gray-900/90 backdrop-blur-xl rounded-xl p-4 border border-cyan-500/30 shadow-2xl">
            <div className="text-xs text-cyan-400 font-mono font-bold mb-3 flex items-center justify-between">
              <span>BENCHMARK METRICS (Table 2)</span>
              <span className="flex items-center gap-1.5">
                <span className={`w-2 h-2 rounded-full ${benchmarkRunning ? 'bg-orange-400 animate-pulse' : 'bg-green-400'}`}></span>
                <span className={benchmarkRunning ? 'text-orange-400' : 'text-green-400'}>
                  {benchmarkRunning ? 'RUNNING' : 'READY'}
                </span>
              </span>
            </div>
            <div className="space-y-2 text-xs font-mono">
              {/* Key benchmark metrics from paper */}
              <div className="flex justify-between gap-6">
                <span className="text-gray-400">Peak Performance:</span>
                <span className="text-cyan-300 font-bold">{BENCHMARK_METRICS.peakPerformance.value} {BENCHMARK_METRICS.peakPerformance.unit}</span>
              </div>
              <div className="flex justify-between gap-6">
                <span className="text-gray-400">Energy Efficiency:</span>
                <span className="text-green-300 font-bold">{BENCHMARK_METRICS.efficiency.value} {BENCHMARK_METRICS.efficiency.unit}</span>
              </div>
              <div className="flex justify-between gap-6">
                <span className="text-gray-400">Tile Utilization:</span>
                <span className="text-cyan-300">{BENCHMARK_METRICS.tileUtilization.value}%</span>
              </div>
              <div className="flex justify-between gap-6">
                <span className="text-gray-400">NoC Bandwidth:</span>
                <span className="text-blue-300">{BENCHMARK_METRICS.nocBandwidth.value} GB/s</span>
              </div>
              <div className="border-t border-gray-700/50 my-2 pt-2"></div>
              <div className="flex justify-between gap-6">
                <span className="text-gray-400">SIS Latency:</span>
                <span className="text-pink-300">{BENCHMARK_METRICS.sisLatency.value} cycles</span>
              </div>
              <div className="flex justify-between gap-6">
                <span className="text-gray-400">Feistel Throughput:</span>
                <span className="text-yellow-300">{BENCHMARK_METRICS.feistelThroughput.value} Gb/s</span>
              </div>
              <div className="flex justify-between gap-6">
                <span className="text-gray-400">Power (Active):</span>
                <span className="text-orange-300">{BENCHMARK_METRICS.powerDissipation.value}W</span>
              </div>
              <div className="border-t border-gray-700/50 my-2 pt-2"></div>
              <div className="text-[10px] text-gray-500">
                Clock: 950MHz (tile) • 1.2GHz (NoC) • Process: N7FF
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Hover Tooltip - Enhanced with tile info */}
      {hoveredComponent && !selectedComponent && (
        <div 
          className="absolute z-20 pointer-events-none"
          style={{ 
            left: lastMouse.x + 20, 
            top: lastMouse.y + 20 
          }}
        >
          <div className="bg-gray-900/95 backdrop-blur-xl rounded-lg p-3 border border-cyan-500/50 shadow-2xl max-w-xs">
            <div className="text-sm font-mono font-bold text-cyan-400 mb-1">
              {EXPLANATIONS[hoveredComponent]?.name || hoveredComponent}
            </div>
            {EXPLANATIONS[hoveredComponent] && (
              <div className="text-xs text-gray-300 mb-2">
                Click to see details
              </div>
            )}
            {activeTile !== null && (
              <div className="border-t border-gray-700 pt-2 mt-2">
                <div className="text-xs font-mono text-gray-400 space-y-1">
                  <div>Tile: <span className="text-cyan-300">T{activeTile.toString().padStart(2, '0')}</span></div>
                  <div>Row: <span className="text-cyan-300">{Math.floor(activeTile / 8)}</span> Col: <span className="text-cyan-300">{activeTile % 8}</span></div>
                  <div>Clock Domain: <span className="text-green-300">clk_tile (950 MHz)</span></div>
                </div>
              </div>
            )}
          </div>
        </div>
      )}

      {/* Right Panel - Collapsible with Tab Navigation */}
      <div className={`transition-all duration-300 ease-out ${uiMinimized ? 'w-12' : 'w-80'} bg-gray-900/95 backdrop-blur-xl border-l border-gray-700/50 flex flex-col overflow-hidden shadow-2xl`}>
        {/* Minimize Toggle */}
        <button
          onClick={() => setUiMinimized(!uiMinimized)}
          className="absolute right-2 top-2 z-10 w-8 h-8 rounded-lg bg-gray-800 border border-gray-700 flex items-center justify-center text-gray-400 hover:text-cyan-400 hover:border-cyan-500/50 transition-all"
          title={uiMinimized ? 'Expand panel' : 'Collapse panel'}
        >
          <span className={`transition-transform duration-300 ${uiMinimized ? 'rotate-180' : ''}`}>◀</span>
        </button>
        
        {!uiMinimized && (
          <>
            {/* Tab Navigation */}
            <div className="flex border-b border-gray-700/50">
              {[
                { id: 'controls', icon: '⚙️', label: 'View' },
                { id: 'components', icon: '🔍', label: 'Parts' },
                { id: 'benchmarks', icon: '🚀', label: 'Bench' }
              ].map(tab => (
                <button
                  key={tab.id}
                  onClick={() => setActivePanel(tab.id)}
                  className={`flex-1 py-3 text-xs font-mono flex flex-col items-center gap-1 transition-all ${
                    activePanel === tab.id
                      ? 'bg-gray-800 text-cyan-400 border-b-2 border-cyan-400'
                      : 'text-gray-500 hover:text-gray-300 hover:bg-gray-800/50'
                  }`}
                >
                  <span className="text-base">{tab.icon}</span>
                  <span>{tab.label}</span>
                </button>
              ))}
            </div>

            {/* Panel Content with smooth transitions */}
            <div className="flex-1 overflow-hidden">
              {/* View Controls Tab */}
              <div className={`h-full overflow-y-auto custom-scrollbar transition-all duration-200 ${activePanel === 'controls' ? 'opacity-100' : 'opacity-0 absolute pointer-events-none'}`}>
                <div className="p-4">
                  <div className="text-sm text-cyan-400 font-mono font-bold mb-3 flex items-center gap-2">
                    <span className="text-lg">⚙</span> VIEW CONTROLS
                  </div>
                  <div className="space-y-3">
                    <div>
                      <label className="text-xs text-gray-400 font-mono mb-1 block">Explode Level</label>
                      <input 
                        type="range" 
                        min="0" 
                        max="2" 
                        step="0.05"
                        value={explodeLevel}
                        onChange={(e) => setExplodeLevel(parseFloat(e.target.value))}
                        className="w-full h-2 bg-gray-700 rounded-lg appearance-none cursor-pointer accent-cyan-500"
                      />
                      <div className="flex justify-between text-xs text-gray-500 font-mono mt-1">
                        <span>0</span>
                        <span>{explodeLevel.toFixed(2)}</span>
                        <span>2</span>
                      </div>
                    </div>
                    <div className="grid grid-cols-3 gap-2">
                      <button
                        onClick={() => setTargetExplode(0)}
                        className="px-2 py-1.5 text-xs font-mono rounded bg-gray-700 text-gray-300 hover:bg-cyan-600 hover:text-white transition-all"
                      >
                        Assembly
                      </button>
                      <button
                        onClick={() => setTargetExplode(1)}
                        className="px-2 py-1.5 text-xs font-mono rounded bg-gray-700 text-gray-300 hover:bg-cyan-600 hover:text-white transition-all"
                      >
                        Explode
                      </button>
                      <button
                        onClick={() => setTargetExplode(2)}
                        className="px-2 py-1.5 text-xs font-mono rounded bg-gray-700 text-gray-300 hover:bg-cyan-600 hover:text-white transition-all"
                      >
                        Full
                      </button>
                    </div>
                    <div className="grid grid-cols-2 gap-2">
                      <button
                        onClick={resetView}
                        className="px-2 py-1.5 text-xs font-mono rounded bg-gray-700 text-gray-300 hover:bg-blue-600 hover:text-white transition-all"
                      >
                        Reset View
                      </button>
                      <button
                        onClick={() => setAutoRotate(!autoRotate)}
                        className={`px-2 py-1.5 text-xs font-mono rounded transition-all ${
                          autoRotate ? 'bg-cyan-600 text-white' : 'bg-gray-700 text-gray-300 hover:bg-cyan-600 hover:text-white'
                        }`}
                      >
                        Auto-Rotate
                      </button>
                    </div>

                    {/* Visual Modes */}
                    <div className="pt-3 border-t border-gray-700/50">
                      <div className="text-xs text-gray-400 font-mono mb-2">Visual Mode</div>
                      <div className="grid grid-cols-2 gap-2">
                        {['normal', 'xray', 'thermal', 'dataflow'].map(mode => (
                          <button
                            key={mode}
                            onClick={() => setVisualMode(mode)}
                            className={`px-2 py-1.5 text-xs font-mono rounded capitalize transition-all ${
                              visualMode === mode 
                                ? 'bg-gradient-to-r from-cyan-600 to-blue-600 text-white shadow-lg' 
                                : 'bg-gray-700 text-gray-300 hover:bg-gray-600'
                            }`}
                          >
                            {mode}
                          </button>
                        ))}
                      </div>
                    </div>

                    {/* Layer Toggles */}
                    <div className="pt-3 border-t border-gray-700/50">
                      <div className="text-xs text-gray-400 font-mono mb-2">Layers</div>
                      <div className="grid grid-cols-2 gap-1.5">
                        {Object.entries(showLayers).map(([key, visible]) => (
                          <button
                            key={key}
                            onClick={() => toggleLayer(key)}
                            className={`px-2 py-1 text-xs font-mono rounded transition-all capitalize ${
                              visible 
                                ? 'bg-cyan-600/80 text-white' 
                                : 'bg-gray-700/50 text-gray-500 hover:bg-gray-600'
                            }`}
                          >
                            {key}
                          </button>
                        ))}
                      </div>
                    </div>

                    {/* Quick Actions */}
                    <div className="pt-3 border-t border-gray-700/50">
                      <button
                        onClick={tourActive ? stopTour : startTour}
                        className={`w-full px-3 py-2.5 text-sm font-mono rounded font-bold transition-all shadow-xl ${
                          tourActive 
                            ? 'bg-gradient-to-r from-red-600 to-orange-600 text-white' 
                            : 'bg-gradient-to-r from-purple-600 to-pink-600 text-white hover:from-purple-500 hover:to-pink-500'
                        }`}
                      >
                        {tourActive ? '⏹ Stop Tour' : '🎬 Guided Tour'}
                      </button>
                    </div>
                  </div>
                </div>
              </div>
              
              {/* Components Tab */}
              <div className={`h-full overflow-y-auto custom-scrollbar transition-all duration-200 ${activePanel === 'components' ? 'opacity-100' : 'opacity-0 absolute pointer-events-none'}`}>
                <div className="p-4">
                  <div className="text-sm text-cyan-400 font-mono font-bold mb-3">🔍 COMPONENTS</div>
                  <div className="space-y-1 mb-4">
                    {Object.keys(EXPLANATIONS).map((key) => (
                      <button
                        key={key}
                        onClick={() => setSelectedComponent(key)}
                        className={`w-full px-3 py-2 text-xs font-mono rounded text-left transition-all ${
                          selectedComponent === key 
                            ? 'bg-gradient-to-r from-cyan-600 to-blue-600 text-white shadow-lg' 
                            : 'bg-gray-800 text-gray-300 hover:bg-gray-700'
                        }`}
                      >
                        {EXPLANATIONS[key].name}
                      </button>
                    ))}
                  </div>
                  
                  {/* Info Display */}
                  {selectedComponent && (
                    <div className="mt-4 pt-4 border-t border-gray-700">
                      <div className="flex items-center justify-between mb-3">
                        <span className="text-xs text-cyan-400 font-mono font-bold">DETAILS</span>
                        <button
                          onClick={() => setSimpleMode(!simpleMode)}
                          className={`px-2 py-1 text-xs font-mono rounded transition-all ${
                            simpleMode 
                              ? 'bg-green-600/80 text-white' 
                              : 'bg-purple-600/80 text-white'
                          }`}
                        >
                          {simpleMode ? '🎓 Simple' : '🔬 Tech'}
                        </button>
                      </div>
                      <div className="text-xs text-gray-300 leading-relaxed bg-gray-800/50 rounded-lg p-3 border border-gray-700">
                        {simpleMode 
                          ? EXPLANATIONS[selectedComponent].simple 
                          : EXPLANATIONS[selectedComponent].technical
                        }
                      </div>
                      {EXPLANATIONS[selectedComponent].specs && (
                        <div className="mt-3 bg-cyan-900/20 border border-cyan-500/30 rounded-lg p-3">
                          <div className="text-xs text-cyan-400 font-mono font-bold mb-1">SPECS</div>
                          <div className="text-xs text-gray-300 font-mono leading-relaxed">
                            {EXPLANATIONS[selectedComponent].specs}
                          </div>
                        </div>
                      )}
                    </div>
                  )}
                </div>
              </div>

              {/* Benchmarks Tab */}
              <div className={`h-full overflow-y-auto custom-scrollbar transition-all duration-200 ${activePanel === 'benchmarks' ? 'opacity-100' : 'opacity-0 absolute pointer-events-none'}`}>
                <div className="p-4">
                  <div className="text-sm text-purple-400 font-mono font-bold mb-3">🚀 BENCHMARKS</div>
                  
                  {/* Benchmark Scenarios */}
                  <div className="space-y-2 mb-4">
                    {Object.entries(BENCHMARK_SCENARIOS).map(([key, scenario]) => (
                      <button
                        key={key}
                        onClick={() => benchmarkRunning ? stopBenchmark() : runBenchmark(scenario.id)}
                        disabled={benchmarkRunning && benchmarkMode !== scenario.id}
                        className={`w-full px-3 py-2 text-xs font-mono rounded transition-all text-left ${
                          benchmarkMode === scenario.id && benchmarkRunning
                            ? 'bg-gradient-to-r from-purple-600 to-pink-600 text-white animate-pulse'
                            : benchmarkRunning 
                              ? 'bg-gray-800 text-gray-500 cursor-not-allowed'
                              : 'bg-gray-800 text-gray-300 hover:bg-gray-700'
                        }`}
                      >
                        <div className="flex items-center justify-between">
                          <span>{scenario.name}</span>
                          {benchmarkMode === scenario.id && benchmarkRunning && (
                            <span className="text-[10px] bg-white/20 px-1 rounded">{benchmarkProgress}%</span>
                          )}
                        </div>
                      </button>
                    ))}
                  </div>

                  {/* Visualization Toggles */}
                  <div className="space-y-2 pt-3 border-t border-gray-700">
                    <div className="text-xs text-gray-400 font-mono mb-2">Visualizations</div>
                    <button
                      onClick={() => setShowThermalView(!showThermalView)}
                      className={`w-full px-3 py-2 text-xs font-mono rounded transition-all ${
                        showThermalView 
                          ? 'bg-gradient-to-r from-orange-600 to-red-600 text-white' 
                          : 'bg-gray-700 text-gray-300 hover:bg-gray-600'
                      }`}
                    >
                      🌡️ Thermal: {showThermalView ? 'ON' : 'OFF'}
                    </button>
                    <button
                      onClick={() => setShowPowerDomains(!showPowerDomains)}
                      className={`w-full px-3 py-2 text-xs font-mono rounded transition-all ${
                        showPowerDomains 
                          ? 'bg-gradient-to-r from-green-600 to-teal-600 text-white' 
                          : 'bg-gray-700 text-gray-300 hover:bg-gray-600'
                      }`}
                    >
                      ⚡ Power: {showPowerDomains ? 'ON' : 'OFF'}
                    </button>
                    <button
                      onClick={() => setShowComparison(!showComparison)}
                      className={`w-full px-3 py-2 text-xs font-mono rounded transition-all ${
                        showComparison 
                          ? 'bg-gradient-to-r from-blue-600 to-indigo-600 text-white' 
                          : 'bg-gray-700 text-gray-300 hover:bg-gray-600'
                      }`}
                    >
                      📈 FFT Compare: {showComparison ? 'ON' : 'OFF'}
                    </button>
                    <button
                      onClick={() => setAnimateNoc(!animateNoc)}
                      className={`w-full px-3 py-2 text-xs font-mono rounded transition-all ${
                        animateNoc 
                          ? 'bg-gradient-to-r from-cyan-600 to-blue-600 text-white' 
                          : 'bg-gray-700 text-gray-300 hover:bg-gray-600'
                      }`}
                    >
                      🌊 NoC Traffic: {animateNoc ? 'ON' : 'OFF'}
                    </button>
                    <button
                      onClick={() => setShowMetrics(!showMetrics)}
                      className={`w-full px-3 py-2 text-xs font-mono rounded transition-all ${
                        showMetrics 
                          ? 'bg-gradient-to-r from-green-600 to-emerald-600 text-white' 
                          : 'bg-gray-700 text-gray-300 hover:bg-gray-600'
                      }`}
                    >
                      📊 Metrics: {showMetrics ? 'ON' : 'OFF'}
                    </button>
                  </div>
                  
                  {/* Clock Domains */}
                  <div className="pt-3 mt-3 border-t border-gray-700">
                    <div className="text-xs text-gray-400 font-mono mb-2">Clock Domains</div>
                    <div className="space-y-1">
                      {Object.entries(CLOCK_DOMAINS).map(([key, domain]) => (
                        <div 
                          key={key}
                          className="flex items-center justify-between px-2 py-1 rounded bg-gray-800/50 text-xs font-mono"
                        >
                          <div className="flex items-center gap-2">
                            <div 
                              className="w-2 h-2 rounded-full animate-pulse"
                              style={{ 
                                backgroundColor: `#${domain.color.toString(16).padStart(6, '0')}`,
                                boxShadow: `0 0 6px #${domain.color.toString(16).padStart(6, '0')}`
                              }}
                            />
                            <span className="text-gray-400">{key}</span>
                          </div>
                          <span className="text-cyan-400">{domain.freq}MHz</span>
                        </div>
                      ))}
                    </div>
                  </div>
                </div>
              </div>
            </div>
          </>
        )}
      </div>

      {/* Comparison Panel (Table 4) */}
      {showComparison && (
        <div className="absolute bottom-6 right-[21rem] z-10">
          <div className="bg-gray-900/95 backdrop-blur-xl rounded-xl p-4 border border-blue-500/30 shadow-2xl">
            <div className="text-xs text-blue-400 font-mono font-bold mb-3">📊 COMPARISON (Table 4)</div>
            <table className="text-xs font-mono">
              <thead>
                <tr className="text-gray-400">
                  <th className="pr-3 text-left">Metric</th>
                  <th className="px-2 text-right text-cyan-300">RFT-Proc</th>
                  <th className="px-2 text-right text-gray-500">FFT Accel</th>
                  <th className="pl-2 text-right text-green-400">Δ</th>
                </tr>
              </thead>
              <tbody className="text-gray-300">
                <tr>
                  <td className="pr-3">TOPS</td>
                  <td className="px-2 text-right text-cyan-300 font-bold">{COMPARISON_DATA.rftpu.tops}</td>
                  <td className="px-2 text-right text-gray-500">{COMPARISON_DATA.traditionalFFT.tops}</td>
                  <td className="pl-2 text-right text-green-400">{COMPARISON_DATA.advantage.ops}</td>
                </tr>
                <tr>
                  <td className="pr-3">GOPS/W</td>
                  <td className="px-2 text-right text-cyan-300 font-bold">{COMPARISON_DATA.rftpu.efficiency}</td>
                  <td className="px-2 text-right text-gray-500">{COMPARISON_DATA.traditionalFFT.efficiency}</td>
                  <td className="pl-2 text-right text-green-400">{COMPARISON_DATA.advantage.efficiency}</td>
                </tr>
                <tr>
                  <td className="pr-3">Area (mm²)</td>
                  <td className="px-2 text-right text-cyan-300 font-bold">{COMPARISON_DATA.rftpu.area}</td>
                  <td className="px-2 text-right text-gray-500">{COMPARISON_DATA.traditionalFFT.area}</td>
                  <td className="pl-2 text-right text-green-400">{COMPARISON_DATA.advantage.area}</td>
                </tr>
              </tbody>
            </table>
          </div>
        </div>
      )}

      {/* Power Domains Panel */}
      {showPowerDomains && (
        <div className="absolute top-32 left-6 z-10">
          <div className="bg-gray-900/95 backdrop-blur-xl rounded-xl p-4 border border-green-500/30 shadow-2xl">
            <div className="text-xs text-green-400 font-mono font-bold mb-3">⚡ POWER DOMAINS</div>
            <div className="space-y-2">
              {Object.entries(POWER_DOMAINS).map(([domain, data]) => (
                <div key={domain} className="flex items-center gap-2">
                  <div 
                    className="w-3 h-3 rounded-full"
                    style={{ 
                      backgroundColor: `#${data.color.toString(16).padStart(6, '0')}`,
                      boxShadow: `0 0 8px #${data.color.toString(16).padStart(6, '0')}`
                    }}
                  />
                  <div className="flex-1 text-xs font-mono">
                    <div className="flex justify-between">
                      <span className="text-gray-300">{data.label}</span>
                      <span className="text-gray-400">{data.voltage}V</span>
                    </div>
                    <div className="text-[10px] text-gray-500">{data.power} mW</div>
                  </div>
                </div>
              ))}
              <div className="border-t border-gray-700 pt-2 mt-2">
                <div className="flex justify-between text-xs font-mono">
                  <span className="text-gray-400">Total Active:</span>
                  <span className="text-orange-300 font-bold">8.2W</span>
                </div>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Thermal Wave Indicator */}
      {showThermalView && (
        <div className="absolute top-32 right-[21rem] z-10">
          <div className="bg-gray-900/95 backdrop-blur-xl rounded-xl p-4 border border-orange-500/30 shadow-2xl">
            <div className="text-xs text-orange-400 font-mono font-bold mb-3">🌡️ THERMAL PROFILE</div>
            <div className="space-y-2">
              <div className="flex items-center gap-2 text-xs font-mono">
                <span className="text-gray-400">Active Wave:</span>
                <span className="text-orange-300 font-bold">{thermalWaveIndex + 1} / 9</span>
              </div>
              <div className="grid grid-cols-9 gap-0.5">
                {ACTIVATION_WAVES.map((wave, idx) => (
                  <div 
                    key={idx}
                    className={`w-4 h-4 rounded-sm transition-all ${
                      idx <= thermalWaveIndex 
                        ? 'bg-gradient-to-br from-orange-500 to-red-600' 
                        : 'bg-gray-700'
                    }`}
                    title={`Wave ${idx + 1}: ${wave.length} tiles`}
                  />
                ))}
              </div>
              <div className="flex justify-between text-[10px] font-mono mt-2">
                {Object.entries(THERMAL_PROFILE).map(([key, data]) => (
                  <div key={key} className="text-center">
                    <div 
                      className="w-3 h-3 rounded-full mx-auto mb-0.5"
                      style={{ backgroundColor: `#${data.color.toString(16).padStart(6, '0')}` }}
                    />
                    <span className="text-gray-500">{data.temp}°C</span>
                  </div>
                ))}
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Enhanced Legend with Paper References */}
      <div className="absolute bottom-6 left-6 z-10">
        <div className="bg-gray-900/90 backdrop-blur-xl rounded-xl p-4 border border-cyan-500/30 shadow-2xl">
          <div className="text-xs text-cyan-400 font-mono font-bold mb-3">RFT-PROCESSOR ARCHITECTURE</div>
          <div className="grid grid-cols-2 gap-x-4 gap-y-2 text-xs font-mono">
            {[
              { color: '#00d4aa', label: 'phi_rft_core', clock: 'clk_tile' },
              { color: '#4a9eff', label: 'rpu_noc_fabric', clock: 'clk_noc' },
              { color: '#ff6b9d', label: 'rft_sis_hash', clock: 'clk_sis' },
              { color: '#ffd700', label: 'feistel_48_cipher', clock: 'clk_feistel' },
              { color: '#9b59b6', label: 'unified_ctrl', clock: null },
              { color: '#e74c3c', label: 'dma_ingress', clock: null }
            ].map(({ color, label, clock }) => (
              <div key={label} className="flex items-center gap-2">
                <div 
                  className="w-4 h-4 rounded shadow-lg"
                  style={{ 
                    background: color,
                    boxShadow: `0 0 10px ${color}40`
                  }}
                ></div>
                <div className="flex flex-col">
                  <span className="text-gray-300">{label}</span>
                  {clock && <span className="text-gray-500 text-[10px]">{clock}</span>}
                </div>
              </div>
            ))}
          </div>
          <div className="mt-3 pt-2 border-t border-gray-700/50 space-y-1">
            <div className="text-[10px] text-gray-400 font-mono">
              📄 {PAPER_REFERENCES.architecture}
            </div>
            <div className="text-[10px] text-gray-400 font-mono">
              📊 {PAPER_REFERENCES.performance}
            </div>
          </div>
        </div>
      </div>

      {/* φ-RFT Pipeline Stage Indicator */}
      {benchmarkMode === 'rft' && benchmarkRunning && (
        <div className="absolute bottom-24 left-6 z-10">
          <div className="bg-gray-900/95 backdrop-blur-xl rounded-xl p-3 border border-cyan-500/30 shadow-2xl">
            <div className="text-xs text-cyan-400 font-mono font-bold mb-2">Ψ = Σ w_i D_φi C_σi D†_φi</div>
            <div className="flex gap-1">
              {RFT_VISUALIZATION.pipelineStages.map((stage, idx) => (
                <div 
                  key={stage}
                  className={`px-2 py-1 rounded text-[10px] font-mono transition-all ${
                    idx === rftStage 
                      ? 'bg-cyan-500 text-white' 
                      : idx < rftStage 
                        ? 'bg-cyan-800 text-cyan-300' 
                        : 'bg-gray-700 text-gray-500'
                  }`}
                >
                  {stage}
                </div>
              ))}
            </div>
          </div>
        </div>
      )}

      {/* Enhanced Instructions */}
      <div className="absolute bottom-6 left-1/2 transform -translate-x-1/2 z-10">
        <div className="bg-gray-900/80 backdrop-blur-xl rounded-full px-6 py-2.5 border border-cyan-500/30 shadow-xl">
          <p className="text-xs text-gray-300 font-mono flex items-center gap-4">
            <span className="flex items-center gap-1.5">
              <span className="text-cyan-400">🖱️</span>
              <span>Drag to rotate</span>
            </span>
            <span className="text-gray-600">•</span>
            <span className="flex items-center gap-1.5">
              <span className="text-cyan-400">🔍</span>
              <span>Scroll to zoom</span>
            </span>
            <span className="text-gray-600">•</span>
            <span className="flex items-center gap-1.5">
              <span className="text-cyan-400">👆</span>
              <span>Click for details</span>
            </span>
            <span className="text-gray-600">•</span>
            <span className="flex items-center gap-1.5">
              <span className="text-cyan-400">⚡</span>
              <span>64 Tiles • φ-RFT Engine</span>
            </span>
          </p>
        </div>
      </div>

      {/* Tour Progress Indicator */}
      {tourActive && (
        <div className="absolute top-6 left-1/2 transform -translate-x-1/2 z-10">
          <div className="bg-purple-900/90 backdrop-blur-xl rounded-full px-6 py-3 border border-purple-500/50 shadow-2xl shadow-purple-500/30">
            <div className="text-sm font-mono text-white flex items-center gap-3">
              <span className="text-lg">🎬</span>
              <span>Guided Tour: Step {tourStep + 1} of {TOUR_STOPS.length}</span>
              <div className="flex gap-1">
                {TOUR_STOPS.map((_, i) => (
                  <div 
                    key={i}
                    className={`w-2 h-2 rounded-full transition-all ${
                      i === tourStep ? 'bg-purple-300 scale-125' : 
                      i < tourStep ? 'bg-purple-500' : 'bg-purple-800'
                    }`}
                  />
                ))}
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Custom Scrollbar Styles & Animations */}
      <style jsx>{`
        .custom-scrollbar::-webkit-scrollbar {
          width: 6px;
        }
        .custom-scrollbar::-webkit-scrollbar-track {
          background: rgba(31, 41, 55, 0.5);
          border-radius: 3px;
        }
        .custom-scrollbar::-webkit-scrollbar-thumb {
          background: rgba(34, 211, 238, 0.5);
          border-radius: 3px;
        }
        .custom-scrollbar::-webkit-scrollbar-thumb:hover {
          background: rgba(34, 211, 238, 0.7);
        }
        @keyframes fade-in {
          from { opacity: 0; transform: translateY(10px); }
          to { opacity: 1; transform: translateY(0); }
        }
        @keyframes slide-in {
          from { transform: translateX(20px); opacity: 0; }
          to { transform: translateX(0); opacity: 1; }
        }
        @keyframes glow-pulse {
          0%, 100% { box-shadow: 0 0 20px rgba(34, 211, 238, 0.3); }
          50% { box-shadow: 0 0 40px rgba(34, 211, 238, 0.6); }
        }
        .animate-fade-in { animation: fade-in 0.3s ease-out; }
        .animate-slide-in { animation: slide-in 0.4s ease-out; }
        .animate-glow-pulse { animation: glow-pulse 2s ease-in-out infinite; }
      `}</style>
    </div>
  );
}