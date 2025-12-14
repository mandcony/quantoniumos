/**
 * SPDX-License-Identifier: LicenseRef-QuantoniumOS-Claims-NC
 * Copyright (C) 2025 Luis M. Minier / quantoniumos
 * 
 * Structural Health Monitor - RFT-Based Vibration Analyzer
 * 
 * Uses the Resonant Fourier Transform (RFT) optimized for structural
 * vibration analysis. The golden-ratio quasi-periodic structure in RFT
 * naturally aligns with structural resonance modes.
 * 
 * Domain Applications:
 * - SEISMIC: Building sway, earthquake response
 * - BRIDGES: Traffic-induced vibration, structural fatigue
 * - CONSTRUCTION: Machinery vibration, pile driving impact
 * - INDUSTRIAL: Rotating equipment, bearing health
 */

import { Complex, abs, multiply, add, conj, exp } from '../rft/Complex';

// Golden ratio - fundamental to structural resonance patterns
const PHI = (1 + Math.sqrt(5)) / 2;
const TWO_PI = 2 * Math.PI;

// ============================================================================
// DOMAIN-SPECIFIC CONFIGURATIONS
// ============================================================================

export type SHMDomain = 'seismic' | 'bridge' | 'construction' | 'industrial' | 'general';

export interface DomainConfig {
  name: string;
  description: string;
  
  // Frequency bands of interest (Hz)
  frequencyBands: {
    name: string;
    low: number;
    high: number;
    significance: string;
  }[];
  
  // RFT configuration
  rftSize: number;
  rftDecayRate: number;    // Higher = faster decay, better for impulsive
  rftF0: number;           // Base frequency for resonance model
  
  // Sampling
  sampleRateHz: number;
  windowDurationSec: number;
  
  // Anomaly detection
  anomalyThreshold: number;  // MAD units
  alertAfterWindows: number;
}

export const DOMAIN_CONFIGS: Record<SHMDomain, DomainConfig> = {
  seismic: {
    name: 'Seismic / Building Sway',
    description: 'Monitor building natural frequency and earthquake response',
    frequencyBands: [
      { name: 'Fundamental', low: 0.1, high: 1.0, significance: 'Building natural sway period' },
      { name: 'Low modes', low: 1.0, high: 5.0, significance: 'Higher structural modes' },
      { name: 'Ground motion', low: 0.5, high: 10.0, significance: 'Earthquake frequency range' },
    ],
    rftSize: 64,
    rftDecayRate: 0.02,   // Slow decay - captures long-period motion
    rftF0: 1.0,           // Low base frequency
    sampleRateHz: 20,
    windowDurationSec: 10,
    anomalyThreshold: 2.5,
    alertAfterWindows: 2,
  },
  
  bridge: {
    name: 'Bridge / Viaduct Monitoring',
    description: 'Track traffic-induced vibration and structural changes',
    frequencyBands: [
      { name: 'Sway modes', low: 0.5, high: 3.0, significance: 'Lateral bridge movement' },
      { name: 'Vertical modes', low: 2.0, high: 10.0, significance: 'Deck vibration' },
      { name: 'Traffic', low: 5.0, high: 25.0, significance: 'Vehicle-induced vibration' },
      { name: 'Cable/stay', low: 1.0, high: 8.0, significance: 'Cable resonance' },
    ],
    rftSize: 128,
    rftDecayRate: 0.03,
    rftF0: 3.0,
    sampleRateHz: 50,
    windowDurationSec: 5,
    anomalyThreshold: 3.0,
    alertAfterWindows: 3,
  },
  
  construction: {
    name: 'Construction Site',
    description: 'Monitor pile driving, excavation, machinery vibration',
    frequencyBands: [
      { name: 'Pile driving', low: 10, high: 50, significance: 'Impact frequency' },
      { name: 'Machinery', low: 20, high: 100, significance: 'Equipment vibration' },
      { name: 'Ground-borne', low: 5, high: 30, significance: 'Propagated vibration' },
    ],
    rftSize: 256,
    rftDecayRate: 0.1,    // Fast decay - impulsive signals
    rftF0: 25.0,
    sampleRateHz: 200,
    windowDurationSec: 2,
    anomalyThreshold: 4.0,
    alertAfterWindows: 1,
  },
  
  industrial: {
    name: 'Industrial Equipment',
    description: 'Rotating machinery, bearing health, motor vibration',
    frequencyBands: [
      { name: 'Shaft speed', low: 10, high: 60, significance: '1x rotation frequency' },
      { name: 'Imbalance', low: 10, high: 60, significance: 'Same as shaft speed' },
      { name: 'Bearing defects', low: 50, high: 500, significance: 'High-frequency impacts' },
      { name: 'Gear mesh', low: 100, high: 1000, significance: 'Teeth engagement frequency' },
    ],
    rftSize: 512,
    rftDecayRate: 0.05,
    rftF0: 50.0,
    sampleRateHz: 1000,
    windowDurationSec: 1,
    anomalyThreshold: 3.5,
    alertAfterWindows: 2,
  },
  
  general: {
    name: 'General Purpose',
    description: 'Balanced settings for unknown applications',
    frequencyBands: [
      { name: 'Low', low: 0.5, high: 5, significance: 'Structural modes' },
      { name: 'Mid', low: 5, high: 50, significance: 'Common vibration' },
      { name: 'High', low: 50, high: 200, significance: 'High frequency' },
    ],
    rftSize: 128,
    rftDecayRate: 0.05,
    rftF0: 10.0,
    sampleRateHz: 100,
    windowDurationSec: 5,
    anomalyThreshold: 3.0,
    alertAfterWindows: 3,
  },
};

// ============================================================================
// RFT KERNEL (Resonant Fourier Transform)
// ============================================================================

/**
 * Build the Resonant Fourier Transform kernel.
 * 
 * This is NOT the φ-phase FFT (which has no sparsity advantage).
 * This is the eigenbasis of the golden quasi-periodic autocorrelation model:
 * 
 *   r[k] = cos(2πf₀k) + cos(2πf₀φk) * exp(-decay*k)
 * 
 * The resulting basis is optimal (in KLT sense) for signals with
 * golden-ratio frequency relationships - which structural resonances often exhibit.
 */
export function buildRFTKernel(N: number, f0: number = 10.0, decayRate: number = 0.05): number[][] {
  // Build autocorrelation model for golden quasi-periodic signals
  const r: number[] = new Array(N);
  
  for (let k = 0; k < N; k++) {
    const t_k = k / N;
    const r_fundamental = Math.cos(TWO_PI * f0 * t_k);
    const r_golden = Math.cos(TWO_PI * f0 * PHI * t_k);
    const decay = Math.exp(-decayRate * k);
    r[k] = (r_fundamental + r_golden) * decay;
  }
  r[0] = 1.0;  // Normalize
  
  // Build Toeplitz autocorrelation matrix
  const R: number[][] = new Array(N);
  for (let i = 0; i < N; i++) {
    R[i] = new Array(N);
    for (let j = 0; j < N; j++) {
      R[i][j] = r[Math.abs(i - j)];
    }
  }
  
  // Compute eigenvectors using power iteration with deflation
  // (Simplified for mobile - not full eigendecomposition)
  const Phi = symmetricEigendecomposition(R, N);
  
  return Phi;
}

/**
 * Simplified eigendecomposition for symmetric matrices
 * Uses power iteration with deflation
 */
function symmetricEigendecomposition(A: number[][], numVectors: number): number[][] {
  const N = A.length;
  const eigenvectors: number[][] = [];
  const workMatrix = A.map(row => [...row]);
  
  for (let v = 0; v < numVectors; v++) {
    // Power iteration to find dominant eigenvector
    let vec = new Array(N).fill(0).map(() => Math.random() - 0.5);
    let norm = Math.sqrt(vec.reduce((s, x) => s + x * x, 0));
    vec = vec.map(x => x / norm);
    
    for (let iter = 0; iter < 50; iter++) {
      // Multiply: newVec = A * vec
      const newVec = new Array(N).fill(0);
      for (let i = 0; i < N; i++) {
        for (let j = 0; j < N; j++) {
          newVec[i] += workMatrix[i][j] * vec[j];
        }
      }
      
      // Normalize
      norm = Math.sqrt(newVec.reduce((s, x) => s + x * x, 0));
      if (norm < 1e-10) break;
      vec = newVec.map(x => x / norm);
    }
    
    eigenvectors.push(vec);
    
    // Deflate: A = A - λ * v * v^T
    // λ = v^T * A * v
    let eigenvalue = 0;
    for (let i = 0; i < N; i++) {
      for (let j = 0; j < N; j++) {
        eigenvalue += vec[i] * workMatrix[i][j] * vec[j];
      }
    }
    
    for (let i = 0; i < N; i++) {
      for (let j = 0; j < N; j++) {
        workMatrix[i][j] -= eigenvalue * vec[i] * vec[j];
      }
    }
  }
  
  return eigenvectors;
}

// ============================================================================
// SHM ANALYZER CLASS
// ============================================================================

export interface SHMFeatures {
  // Time domain
  rmsVibration: number;
  peakVibration: number;
  crestFactor: number;
  
  // Frequency domain (RFT-based)
  dominantFrequency: number;
  rftEnergy: number[];           // Energy in each band
  rftSparsity: number;           // How concentrated the energy is
  spectralCentroid: number;
  
  // Golden-ratio specific
  phiRatioStrength: number;      // Strength of φ-related frequencies
  resonanceScore: number;        // Match to expected structural resonance
}

export interface SHMAlert {
  type: 'info' | 'warning' | 'critical';
  message: string;
  feature: string;
  value: number;
  threshold: number;
  timestamp: number;
}

export class SHMAnalyzer {
  private domain: SHMDomain;
  private config: DomainConfig;
  private rftKernel: number[][] | null = null;
  
  // Baseline tracking
  private baselineFeatures: SHMFeatures | null = null;
  private baselineMAD: Partial<SHMFeatures> = {};
  private consecutiveAnomalies: number = 0;
  
  // History
  private featureHistory: SHMFeatures[] = [];
  private alerts: SHMAlert[] = [];
  
  constructor(domain: SHMDomain = 'general') {
    this.domain = domain;
    this.config = DOMAIN_CONFIGS[domain];
  }
  
  /**
   * Switch domain configuration
   */
  setDomain(domain: SHMDomain): void {
    this.domain = domain;
    this.config = DOMAIN_CONFIGS[domain];
    this.rftKernel = null;  // Force rebuild
    this.resetBaseline();
  }
  
  /**
   * Get RFT kernel (lazy-loaded)
   */
  private getRFTKernel(): number[][] {
    if (!this.rftKernel) {
      this.rftKernel = buildRFTKernel(
        this.config.rftSize,
        this.config.rftF0,
        this.config.rftDecayRate
      );
    }
    return this.rftKernel;
  }
  
  /**
   * Apply RFT transform
   */
  applyRFT(signal: number[]): number[] {
    const kernel = this.getRFTKernel();
    const N = kernel.length;
    
    // Pad or truncate signal
    const padded = new Array(N).fill(0);
    for (let i = 0; i < Math.min(signal.length, N); i++) {
      padded[i] = signal[i];
    }
    
    // Transform: X = Φ^T * x
    const result = new Array(N).fill(0);
    for (let k = 0; k < N; k++) {
      for (let n = 0; n < N; n++) {
        result[k] += kernel[k][n] * padded[n];
      }
    }
    
    return result;
  }
  
  /**
   * Extract features from accelerometer window
   */
  extractFeatures(samples: { x: number; y: number; z: number }[]): SHMFeatures {
    // Compute magnitude signal (remove gravity approximation)
    const magnitude = samples.map(s => {
      const mag = Math.sqrt(s.x * s.x + s.y * s.y + s.z * s.z);
      return mag - 1.0;  // Subtract gravity (~1g in expo-sensors units)
    });
    
    // Time-domain features
    const rmsVibration = Math.sqrt(magnitude.reduce((s, x) => s + x * x, 0) / magnitude.length);
    const peakVibration = Math.max(...magnitude.map(Math.abs));
    const crestFactor = peakVibration / (rmsVibration + 1e-10);
    
    // Apply RFT
    const rftCoeffs = this.applyRFT(magnitude);
    
    // Compute energy in each frequency band
    const rftEnergy = this.computeBandEnergies(rftCoeffs);
    
    // Sparsity: how many coefficients for 90% energy
    const rftSparsity = this.computeSparsity(rftCoeffs, 0.9);
    
    // Spectral centroid
    const totalEnergy = rftCoeffs.reduce((s, x) => s + x * x, 0);
    let spectralCentroid = 0;
    for (let k = 0; k < rftCoeffs.length; k++) {
      spectralCentroid += k * rftCoeffs[k] * rftCoeffs[k];
    }
    spectralCentroid /= (totalEnergy + 1e-10);
    
    // Dominant frequency
    let maxEnergy = 0;
    let dominantIdx = 0;
    for (let k = 0; k < rftCoeffs.length; k++) {
      const energy = rftCoeffs[k] * rftCoeffs[k];
      if (energy > maxEnergy) {
        maxEnergy = energy;
        dominantIdx = k;
      }
    }
    const dominantFrequency = (dominantIdx * this.config.sampleRateHz) / (2 * rftCoeffs.length);
    
    // Golden-ratio specific: check for φ-related frequency pairs
    const phiRatioStrength = this.computePhiRatioStrength(rftCoeffs);
    
    // Resonance score: how well the signal matches expected structural pattern
    const resonanceScore = this.computeResonanceScore(rftCoeffs);
    
    const features: SHMFeatures = {
      rmsVibration,
      peakVibration,
      crestFactor,
      dominantFrequency,
      rftEnergy,
      rftSparsity,
      spectralCentroid,
      phiRatioStrength,
      resonanceScore,
    };
    
    this.featureHistory.push(features);
    if (this.featureHistory.length > 100) {
      this.featureHistory.shift();
    }
    
    return features;
  }
  
  /**
   * Compute energy in domain-specific frequency bands
   */
  private computeBandEnergies(rftCoeffs: number[]): number[] {
    const N = rftCoeffs.length;
    const sampleRate = this.config.sampleRateHz;
    const bands = this.config.frequencyBands;
    
    const energies: number[] = [];
    
    for (const band of bands) {
      let energy = 0;
      for (let k = 0; k < N; k++) {
        const freq = (k * sampleRate) / (2 * N);
        if (freq >= band.low && freq <= band.high) {
          energy += rftCoeffs[k] * rftCoeffs[k];
        }
      }
      energies.push(energy);
    }
    
    return energies;
  }
  
  /**
   * Compute sparsity: fraction of coefficients for threshold energy
   */
  private computeSparsity(coeffs: number[], threshold: number): number {
    const energies = coeffs.map(x => x * x);
    const totalEnergy = energies.reduce((a, b) => a + b, 0);
    
    const sorted = [...energies].sort((a, b) => b - a);
    
    let cumulative = 0;
    let count = 0;
    for (const e of sorted) {
      cumulative += e;
      count++;
      if (cumulative >= threshold * totalEnergy) break;
    }
    
    return count / coeffs.length;
  }
  
  /**
   * Check for golden-ratio frequency relationships
   * Structural resonances often exhibit φ-related mode spacing
   */
  private computePhiRatioStrength(coeffs: number[]): number {
    const N = coeffs.length;
    let phiScore = 0;
    let pairs = 0;
    
    // Look for coefficient pairs at ratio ~φ
    for (let k1 = 1; k1 < N / 2; k1++) {
      const k2 = Math.round(k1 * PHI);
      if (k2 < N) {
        const e1 = coeffs[k1] * coeffs[k1];
        const e2 = coeffs[k2] * coeffs[k2];
        if (e1 > 0.01 && e2 > 0.01) {
          phiScore += Math.sqrt(e1 * e2);
          pairs++;
        }
      }
    }
    
    return pairs > 0 ? phiScore / pairs : 0;
  }
  
  /**
   * Compute how well signal matches structural resonance pattern
   */
  private computeResonanceScore(coeffs: number[]): number {
    // Expected: energy concentrated in low modes with decay
    const N = coeffs.length;
    const energies = coeffs.map(x => x * x);
    const totalEnergy = energies.reduce((a, b) => a + b, 0) + 1e-10;
    
    // Weight: prefer energy in lower frequencies (structural modes)
    let weightedSum = 0;
    for (let k = 0; k < N; k++) {
      const weight = Math.exp(-k / (N * 0.3));  // Exponential decay
      weightedSum += energies[k] * weight;
    }
    
    return weightedSum / totalEnergy;
  }
  
  /**
   * Set baseline from collected windows
   */
  setBaseline(windows: SHMFeatures[]): void {
    if (windows.length < 3) return;
    
    // Compute median for each feature
    const median = (arr: number[]): number => {
      const sorted = [...arr].sort((a, b) => a - b);
      const mid = Math.floor(sorted.length / 2);
      return sorted.length % 2 ? sorted[mid] : (sorted[mid - 1] + sorted[mid]) / 2;
    };
    
    const mad = (arr: number[], med: number): number => {
      return median(arr.map(x => Math.abs(x - med)));
    };
    
    const rmsValues = windows.map(w => w.rmsVibration);
    const peakValues = windows.map(w => w.peakVibration);
    const crestValues = windows.map(w => w.crestFactor);
    const domFreqValues = windows.map(w => w.dominantFrequency);
    const sparsityValues = windows.map(w => w.rftSparsity);
    const centroidValues = windows.map(w => w.spectralCentroid);
    
    this.baselineFeatures = {
      rmsVibration: median(rmsValues),
      peakVibration: median(peakValues),
      crestFactor: median(crestValues),
      dominantFrequency: median(domFreqValues),
      rftEnergy: [],  // Simplified for now
      rftSparsity: median(sparsityValues),
      spectralCentroid: median(centroidValues),
      phiRatioStrength: median(windows.map(w => w.phiRatioStrength)),
      resonanceScore: median(windows.map(w => w.resonanceScore)),
    };
    
    this.baselineMAD = {
      rmsVibration: mad(rmsValues, this.baselineFeatures.rmsVibration),
      peakVibration: mad(peakValues, this.baselineFeatures.peakVibration),
      crestFactor: mad(crestValues, this.baselineFeatures.crestFactor),
      dominantFrequency: mad(domFreqValues, this.baselineFeatures.dominantFrequency),
      rftSparsity: mad(sparsityValues, this.baselineFeatures.rftSparsity),
      spectralCentroid: mad(centroidValues, this.baselineFeatures.spectralCentroid),
    };
  }
  
  /**
   * Check current features against baseline
   */
  checkAnomaly(features: SHMFeatures): { isAnomaly: boolean; score: number; alerts: SHMAlert[] } {
    if (!this.baselineFeatures) {
      return { isAnomaly: false, score: 0, alerts: [] };
    }
    
    const threshold = this.config.anomalyThreshold;
    const newAlerts: SHMAlert[] = [];
    let maxScore = 0;
    
    const checkFeature = (name: keyof SHMFeatures, current: number, baseline: number, madVal: number) => {
      if (madVal === undefined || madVal < 1e-10) return 0;
      const deviation = Math.abs(current - baseline) / madVal;
      if (deviation > maxScore) maxScore = deviation;
      
      if (deviation > threshold) {
        newAlerts.push({
          type: deviation > threshold * 2 ? 'critical' : 'warning',
          message: `${name}: ${current.toFixed(3)} (baseline: ${baseline.toFixed(3)})`,
          feature: name,
          value: current,
          threshold: baseline + threshold * madVal,
          timestamp: Date.now(),
        });
      }
      return deviation;
    };
    
    checkFeature('rmsVibration', features.rmsVibration, 
      this.baselineFeatures.rmsVibration, this.baselineMAD.rmsVibration || 0.01);
    checkFeature('peakVibration', features.peakVibration,
      this.baselineFeatures.peakVibration, this.baselineMAD.peakVibration || 0.01);
    checkFeature('dominantFrequency', features.dominantFrequency,
      this.baselineFeatures.dominantFrequency, this.baselineMAD.dominantFrequency || 1);
    checkFeature('rftSparsity', features.rftSparsity,
      this.baselineFeatures.rftSparsity, this.baselineMAD.rftSparsity || 0.05);
    
    const isAnomaly = newAlerts.length > 0;
    
    if (isAnomaly) {
      this.consecutiveAnomalies++;
    } else {
      this.consecutiveAnomalies = 0;
    }
    
    // Only alert after consecutive anomalies
    if (this.consecutiveAnomalies >= this.config.alertAfterWindows) {
      this.alerts.push(...newAlerts);
    }
    
    return { 
      isAnomaly, 
      score: maxScore, 
      alerts: this.consecutiveAnomalies >= this.config.alertAfterWindows ? newAlerts : [] 
    };
  }
  
  /**
   * Reset baseline
   */
  resetBaseline(): void {
    this.baselineFeatures = null;
    this.baselineMAD = {};
    this.consecutiveAnomalies = 0;
    this.featureHistory = [];
    this.alerts = [];
  }
  
  /**
   * Get configuration for current domain
   */
  getConfig(): DomainConfig {
    return this.config;
  }
  
  /**
   * Get all alerts
   */
  getAlerts(): SHMAlert[] {
    return [...this.alerts];
  }
  
  /**
   * Get feature history
   */
  getHistory(): SHMFeatures[] {
    return [...this.featureHistory];
  }
  
  /**
   * Check if baseline is set
   */
  hasBaseline(): boolean {
    return this.baselineFeatures !== null;
  }
  
  /**
   * Export session data for analysis
   */
  exportSession(): object {
    return {
      domain: this.domain,
      config: this.config,
      baseline: this.baselineFeatures,
      baselineMAD: this.baselineMAD,
      history: this.featureHistory,
      alerts: this.alerts,
      exportedAt: new Date().toISOString(),
    };
  }
}
