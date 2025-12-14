/**
 * SPDX-License-Identifier: LicenseRef-QuantoniumOS-Claims-NC
 * Copyright (C) 2025 Luis M. Minier / quantoniumos
 * 
 * Structural Health Monitor - Vibration Analyzer
 * 
 * Uses iPhone accelerometer + Φ-RFT to detect structural changes over time
 * by tracking modal/vibration signature drift.
 * 
 * What this CAN do reliably:
 * - Measure ambient/forced vibration response
 * - Track dominant frequencies, band energy, spectral entropy, coherence
 * - Detect change points relative to baseline (unsupervised anomaly detection)
 * 
 * What this CANNOT claim:
 * - Exact damage location (requires sensor arrays / FEM model)
 * - Engineering-grade certification (phones vary, mounting matters)
 */

import { Complex, abs, multiply, add, conj } from '../rft/Complex';
import { CanonicalTrueRFT } from '../rft/RFTCore';

// Golden ratio for Φ-RFT
const PHI = (1 + Math.sqrt(5)) / 2;

// ============================================================================
// TYPES
// ============================================================================

export interface AccelerometerSample {
  timestamp: number;
  x: number;
  y: number;
  z: number;
  magnitude: number;
}

export interface VibrationWindow {
  id: string;
  startTime: number;
  endTime: number;
  samples: AccelerometerSample[];
  sampleRate: number;
}

export interface SpectralFeatures {
  // Peak frequencies and amplitudes (top 5)
  peakFrequencies: number[];
  peakAmplitudes: number[];
  
  // Band energies (Hz ranges)
  bandEnergies: {
    veryLow: number;    // 0.5-2 Hz (building sway)
    low: number;        // 2-10 Hz (structural modes)
    mid: number;        // 10-30 Hz (machinery/traffic)
    high: number;       // 30-50 Hz (high-freq vibration)
  };
  
  // Statistical features
  spectralCentroid: number;      // "center of mass" of spectrum
  spectralRolloff: number;       // frequency below which 85% energy
  spectralEntropy: number;       // disorder/complexity of spectrum
  spectralFlatness: number;      // tonal vs noise-like
  
  // Time-domain features
  crestFactor: number;           // peak/RMS ratio (impulsiveness)
  kurtosis: number;              // "tailedness" (impulse detection)
  rmsLevel: number;              // overall vibration level
  
  // Φ-RFT specific
  rftSparsity: number;           // coefficients for 99% energy
  rftCoherence: number;          // coherence with golden-ratio structure
  dominantRFTMode: number;       // strongest RFT coefficient index
}

export interface BaselineModel {
  id: string;
  name: string;
  createdAt: number;
  updatedAt: number;
  windowCount: number;
  
  // Feature statistics (robust: median + MAD)
  featureMedians: SpectralFeatures;
  featureMADs: SpectralFeatures;   // Median Absolute Deviation
  
  // Raw baseline spectra for coherence
  baselineFFTMagnitude: number[];
  baselineRFTMagnitude: number[];
}

export interface AnomalyScore {
  timestamp: number;
  overallScore: number;           // 0-1, higher = more anomalous
  featureScores: Partial<Record<keyof SpectralFeatures, number>>;
  isAnomaly: boolean;
  consecutiveAnomalies: number;
  triggeredFeatures: string[];
}

export interface SHMEvent {
  id: string;
  timestamp: number;
  type: 'baseline_created' | 'anomaly_detected' | 'baseline_updated' | 'manual_snapshot';
  score: number;
  features: SpectralFeatures;
  waveformSnapshot: number[];     // Raw accelerometer data
  rftCoefficients: Complex[];     // Φ-RFT coefficients
  notes?: string;
}

// ============================================================================
// CONFIGURATION
// ============================================================================

export interface SHMConfig {
  // Window settings
  windowDurationMs: number;       // e.g., 10000 (10 seconds)
  windowOverlap: number;          // e.g., 0.5 (50% overlap)
  
  // Preprocessing
  highpassCutoffHz: number;       // e.g., 0.5 Hz
  lowpassCutoffHz: number;        // e.g., 50 Hz
  removeGravity: boolean;         // Use userAcceleration if available
  
  // Analysis
  fftSize: number;                // e.g., 512, 1024, 2048
  rftSize: number;                // Must match fftSize for comparison
  
  // Anomaly detection
  anomalyThreshold: number;       // e.g., 3.0 (MAD units)
  consecutiveWindowsForAlert: number;  // e.g., 3
  
  // Baseline
  baselineWindowCount: number;    // Windows to average for baseline
  baselineUpdateRate: number;     // EWMA alpha for rolling baseline
}

export const DEFAULT_SHM_CONFIG: SHMConfig = {
  windowDurationMs: 5000,
  windowOverlap: 0.25,
  highpassCutoffHz: 0.5,
  lowpassCutoffHz: 50,
  removeGravity: true,
  fftSize: 128,
  rftSize: 64,  // Small size for mobile performance
  anomalyThreshold: 3.0,
  consecutiveWindowsForAlert: 3,
  baselineWindowCount: 5,
  baselineUpdateRate: 0.1,
};

// ============================================================================
// VIBRATION ANALYZER CLASS
// ============================================================================

export class VibrationAnalyzer {
  private config: SHMConfig;
  private rft: CanonicalTrueRFT | null = null;  // Lazy-loaded
  private baseline: BaselineModel | null = null;
  private consecutiveAnomalies: number = 0;
  private events: SHMEvent[] = [];
  
  constructor(config: Partial<SHMConfig> = {}) {
    this.config = { ...DEFAULT_SHM_CONFIG, ...config };
    // RFT is lazy-loaded on first use to avoid blocking screen load
  }
  
  /**
   * Get RFT instance (lazy-loaded)
   */
  private getRFT(): CanonicalTrueRFT {
    if (!this.rft) {
      this.rft = new CanonicalTrueRFT(this.config.rftSize);
    }
    return this.rft;
  }
  
  // ==========================================================================
  // PREPROCESSING
  // ==========================================================================
  
  /**
   * Remove DC offset and apply bandpass filter
   */
  preprocessSignal(samples: AccelerometerSample[]): number[] {
    const signal = samples.map(s => s.magnitude);
    
    // Remove DC (mean)
    const mean = signal.reduce((a, b) => a + b, 0) / signal.length;
    let filtered = signal.map(x => x - mean);
    
    // Simple highpass: subtract exponential moving average
    if (this.config.highpassCutoffHz > 0) {
      const sampleRate = samples.length / 
        ((samples[samples.length - 1].timestamp - samples[0].timestamp) / 1000);
      const alpha = 2 * Math.PI * this.config.highpassCutoffHz / sampleRate;
      let ema = filtered[0];
      filtered = filtered.map(x => {
        ema = alpha * x + (1 - alpha) * ema;
        return x - ema;
      });
    }
    
    return filtered;
  }
  
  /**
   * Compute magnitude from xyz components
   */
  computeMagnitude(x: number, y: number, z: number): number {
    return Math.sqrt(x * x + y * y + z * z);
  }
  
  // ==========================================================================
  // FFT (Standard comparison baseline)
  // ==========================================================================
  
  /**
   * Compute FFT using Cooley-Tukey radix-2
   */
  computeFFT(signal: number[]): Complex[] {
    const N = signal.length;
    
    // Pad to power of 2
    const size = Math.pow(2, Math.ceil(Math.log2(N)));
    const padded = [...signal];
    while (padded.length < size) padded.push(0);
    
    // Bit-reversal permutation
    const output: Complex[] = padded.map(x => ({ re: x, im: 0 }));
    const bits = Math.log2(size);
    
    for (let i = 0; i < size; i++) {
      const j = this.reverseBits(i, bits);
      if (j > i) {
        [output[i], output[j]] = [output[j], output[i]];
      }
    }
    
    // Cooley-Tukey FFT
    for (let len = 2; len <= size; len *= 2) {
      const ang = -2 * Math.PI / len;
      const wlen: Complex = { re: Math.cos(ang), im: Math.sin(ang) };
      
      for (let i = 0; i < size; i += len) {
        let w: Complex = { re: 1, im: 0 };
        
        for (let j = 0; j < len / 2; j++) {
          const u = output[i + j];
          const v = multiply(output[i + j + len / 2], w);
          
          output[i + j] = add(u, v);
          output[i + j + len / 2] = {
            re: u.re - v.re,
            im: u.im - v.im
          };
          
          w = multiply(w, wlen);
        }
      }
    }
    
    return output;
  }
  
  private reverseBits(n: number, bits: number): number {
    let result = 0;
    for (let i = 0; i < bits; i++) {
      result = (result << 1) | (n & 1);
      n >>= 1;
    }
    return result;
  }
  
  // ==========================================================================
  // Φ-RFT ANALYSIS
  // ==========================================================================
  
  /**
   * Compute Φ-RFT using the canonical implementation
   */
  computeRFT(signal: number[]): Complex[] {
    // Pad/truncate to RFT size
    const input: Complex[] = [];
    for (let i = 0; i < this.config.rftSize; i++) {
      input.push({
        re: i < signal.length ? signal[i] : 0,
        im: 0
      });
    }
    
    return this.getRFT().forwardTransform(input);
  }
  
  /**
   * Compute sparsity: number of coefficients for 99% energy
   */
  computeSparsity(coefficients: Complex[], threshold: number = 0.99): number {
    const magnitudes = coefficients.map(c => abs(c) ** 2);
    const totalEnergy = magnitudes.reduce((a, b) => a + b, 0);
    
    // Sort descending
    const sorted = [...magnitudes].sort((a, b) => b - a);
    
    let cumulative = 0;
    let count = 0;
    for (const mag of sorted) {
      cumulative += mag;
      count++;
      if (cumulative >= threshold * totalEnergy) break;
    }
    
    return count / coefficients.length; // Normalized sparsity ratio
  }
  
  /**
   * Compute coherence with baseline spectrum
   */
  computeCoherence(current: number[], baseline: number[]): number {
    if (current.length !== baseline.length) return 0;
    
    // Cross-correlation normalized
    const N = current.length;
    let dotProduct = 0;
    let normCurrent = 0;
    let normBaseline = 0;
    
    for (let i = 0; i < N; i++) {
      dotProduct += current[i] * baseline[i];
      normCurrent += current[i] * current[i];
      normBaseline += baseline[i] * baseline[i];
    }
    
    const denom = Math.sqrt(normCurrent * normBaseline);
    return denom > 0 ? dotProduct / denom : 0;
  }
  
  // ==========================================================================
  // FEATURE EXTRACTION
  // ==========================================================================
  
  /**
   * Extract all spectral features from a vibration window
   */
  extractFeatures(window: VibrationWindow): SpectralFeatures {
    const signal = this.preprocessSignal(window.samples);
    const sampleRate = window.sampleRate;
    
    // Compute transforms
    const fftCoeffs = this.computeFFT(signal);
    const rftCoeffs = this.computeRFT(signal);
    
    const fftMagnitudes = fftCoeffs.slice(0, fftCoeffs.length / 2).map(c => abs(c));
    const rftMagnitudes = rftCoeffs.map(c => abs(c));
    
    // Frequency bins
    const freqResolution = sampleRate / fftCoeffs.length;
    const frequencies = fftMagnitudes.map((_, i) => i * freqResolution);
    
    // Peak detection (top 5)
    const indexed = fftMagnitudes.map((m, i) => ({ mag: m, freq: frequencies[i], idx: i }));
    indexed.sort((a, b) => b.mag - a.mag);
    const peaks = indexed.slice(0, 5);
    
    // Band energies
    const bandEnergies = {
      veryLow: this.computeBandEnergy(fftMagnitudes, frequencies, 0.5, 2),
      low: this.computeBandEnergy(fftMagnitudes, frequencies, 2, 10),
      mid: this.computeBandEnergy(fftMagnitudes, frequencies, 10, 30),
      high: this.computeBandEnergy(fftMagnitudes, frequencies, 30, 50),
    };
    
    // Spectral statistics
    const totalEnergy = fftMagnitudes.reduce((a, b) => a + b * b, 0);
    
    // Spectral centroid
    let weightedSum = 0;
    let sumMag = 0;
    for (let i = 0; i < fftMagnitudes.length; i++) {
      weightedSum += frequencies[i] * fftMagnitudes[i];
      sumMag += fftMagnitudes[i];
    }
    const spectralCentroid = sumMag > 0 ? weightedSum / sumMag : 0;
    
    // Spectral rolloff (85% energy)
    let cumEnergy = 0;
    let rolloffIdx = 0;
    for (let i = 0; i < fftMagnitudes.length; i++) {
      cumEnergy += fftMagnitudes[i] ** 2;
      if (cumEnergy >= 0.85 * totalEnergy) {
        rolloffIdx = i;
        break;
      }
    }
    const spectralRolloff = frequencies[rolloffIdx] || 0;
    
    // Spectral entropy
    const normalizedMags = fftMagnitudes.map(m => m / (sumMag || 1));
    let entropy = 0;
    for (const p of normalizedMags) {
      if (p > 0) entropy -= p * Math.log2(p);
    }
    const spectralEntropy = entropy / Math.log2(fftMagnitudes.length); // Normalized
    
    // Spectral flatness (geometric mean / arithmetic mean)
    const logSum = fftMagnitudes.reduce((a, m) => a + Math.log(m + 1e-10), 0);
    const geometricMean = Math.exp(logSum / fftMagnitudes.length);
    const arithmeticMean = sumMag / fftMagnitudes.length;
    const spectralFlatness = arithmeticMean > 0 ? geometricMean / arithmeticMean : 0;
    
    // Time-domain features
    const rms = Math.sqrt(signal.reduce((a, x) => a + x * x, 0) / signal.length);
    const peak = Math.max(...signal.map(Math.abs));
    const crestFactor = rms > 0 ? peak / rms : 0;
    
    // Kurtosis
    const mean = signal.reduce((a, b) => a + b, 0) / signal.length;
    const m2 = signal.reduce((a, x) => a + (x - mean) ** 2, 0) / signal.length;
    const m4 = signal.reduce((a, x) => a + (x - mean) ** 4, 0) / signal.length;
    const kurtosis = m2 > 0 ? m4 / (m2 ** 2) - 3 : 0; // Excess kurtosis
    
    // Φ-RFT specific features
    const rftSparsity = this.computeSparsity(rftCoeffs);
    const rftCoherence = this.baseline 
      ? this.computeCoherence(rftMagnitudes, this.baseline.baselineRFTMagnitude)
      : 1.0;
    
    const dominantRFTMode = rftMagnitudes.indexOf(Math.max(...rftMagnitudes));
    
    return {
      peakFrequencies: peaks.map(p => p.freq),
      peakAmplitudes: peaks.map(p => p.mag),
      bandEnergies,
      spectralCentroid,
      spectralRolloff,
      spectralEntropy,
      spectralFlatness,
      crestFactor,
      kurtosis,
      rmsLevel: rms,
      rftSparsity,
      rftCoherence,
      dominantRFTMode,
    };
  }
  
  private computeBandEnergy(
    magnitudes: number[], 
    frequencies: number[], 
    lowHz: number, 
    highHz: number
  ): number {
    let energy = 0;
    for (let i = 0; i < magnitudes.length; i++) {
      if (frequencies[i] >= lowHz && frequencies[i] < highHz) {
        energy += magnitudes[i] ** 2;
      }
    }
    return energy;
  }
  
  // ==========================================================================
  // BASELINE MANAGEMENT
  // ==========================================================================
  
  /**
   * Create a new baseline from multiple windows
   */
  createBaseline(windows: VibrationWindow[], name: string = 'default'): BaselineModel {
    const allFeatures: SpectralFeatures[] = windows.map(w => this.extractFeatures(w));
    
    // Compute medians for each feature
    const featureMedians = this.computeFeatureMedians(allFeatures);
    const featureMADs = this.computeFeatureMADs(allFeatures, featureMedians);
    
    // Average spectra for coherence reference
    const fftMagnitudes: number[][] = [];
    const rftMagnitudes: number[][] = [];
    
    for (const window of windows) {
      const signal = this.preprocessSignal(window.samples);
      const fft = this.computeFFT(signal);
      const rft = this.computeRFT(signal);
      
      fftMagnitudes.push(fft.slice(0, fft.length / 2).map(c => abs(c)));
      rftMagnitudes.push(rft.map(c => abs(c)));
    }
    
    const avgFFT = this.averageArrays(fftMagnitudes);
    const avgRFT = this.averageArrays(rftMagnitudes);
    
    this.baseline = {
      id: `baseline_${Date.now()}`,
      name,
      createdAt: Date.now(),
      updatedAt: Date.now(),
      windowCount: windows.length,
      featureMedians,
      featureMADs,
      baselineFFTMagnitude: avgFFT,
      baselineRFTMagnitude: avgRFT,
    };
    
    // Log event
    this.events.push({
      id: `event_${Date.now()}`,
      timestamp: Date.now(),
      type: 'baseline_created',
      score: 0,
      features: featureMedians,
      waveformSnapshot: [],
      rftCoefficients: [],
      notes: `Baseline "${name}" created from ${windows.length} windows`,
    });
    
    return this.baseline;
  }
  
  private computeFeatureMedians(features: SpectralFeatures[]): SpectralFeatures {
    const median = (arr: number[]) => {
      const sorted = [...arr].sort((a, b) => a - b);
      const mid = Math.floor(sorted.length / 2);
      return sorted.length % 2 ? sorted[mid] : (sorted[mid - 1] + sorted[mid]) / 2;
    };
    
    return {
      peakFrequencies: features[0].peakFrequencies.map((_, i) => 
        median(features.map(f => f.peakFrequencies[i] || 0))),
      peakAmplitudes: features[0].peakAmplitudes.map((_, i) => 
        median(features.map(f => f.peakAmplitudes[i] || 0))),
      bandEnergies: {
        veryLow: median(features.map(f => f.bandEnergies.veryLow)),
        low: median(features.map(f => f.bandEnergies.low)),
        mid: median(features.map(f => f.bandEnergies.mid)),
        high: median(features.map(f => f.bandEnergies.high)),
      },
      spectralCentroid: median(features.map(f => f.spectralCentroid)),
      spectralRolloff: median(features.map(f => f.spectralRolloff)),
      spectralEntropy: median(features.map(f => f.spectralEntropy)),
      spectralFlatness: median(features.map(f => f.spectralFlatness)),
      crestFactor: median(features.map(f => f.crestFactor)),
      kurtosis: median(features.map(f => f.kurtosis)),
      rmsLevel: median(features.map(f => f.rmsLevel)),
      rftSparsity: median(features.map(f => f.rftSparsity)),
      rftCoherence: median(features.map(f => f.rftCoherence)),
      dominantRFTMode: median(features.map(f => f.dominantRFTMode)),
    };
  }
  
  private computeFeatureMADs(
    features: SpectralFeatures[], 
    medians: SpectralFeatures
  ): SpectralFeatures {
    const mad = (arr: number[], med: number) => {
      const deviations = arr.map(x => Math.abs(x - med));
      const sorted = deviations.sort((a, b) => a - b);
      const mid = Math.floor(sorted.length / 2);
      return sorted.length % 2 ? sorted[mid] : (sorted[mid - 1] + sorted[mid]) / 2;
    };
    
    return {
      peakFrequencies: medians.peakFrequencies.map((m, i) => 
        mad(features.map(f => f.peakFrequencies[i] || 0), m)),
      peakAmplitudes: medians.peakAmplitudes.map((m, i) => 
        mad(features.map(f => f.peakAmplitudes[i] || 0), m)),
      bandEnergies: {
        veryLow: mad(features.map(f => f.bandEnergies.veryLow), medians.bandEnergies.veryLow),
        low: mad(features.map(f => f.bandEnergies.low), medians.bandEnergies.low),
        mid: mad(features.map(f => f.bandEnergies.mid), medians.bandEnergies.mid),
        high: mad(features.map(f => f.bandEnergies.high), medians.bandEnergies.high),
      },
      spectralCentroid: mad(features.map(f => f.spectralCentroid), medians.spectralCentroid),
      spectralRolloff: mad(features.map(f => f.spectralRolloff), medians.spectralRolloff),
      spectralEntropy: mad(features.map(f => f.spectralEntropy), medians.spectralEntropy),
      spectralFlatness: mad(features.map(f => f.spectralFlatness), medians.spectralFlatness),
      crestFactor: mad(features.map(f => f.crestFactor), medians.crestFactor),
      kurtosis: mad(features.map(f => f.kurtosis), medians.kurtosis),
      rmsLevel: mad(features.map(f => f.rmsLevel), medians.rmsLevel),
      rftSparsity: mad(features.map(f => f.rftSparsity), medians.rftSparsity),
      rftCoherence: mad(features.map(f => f.rftCoherence), medians.rftCoherence),
      dominantRFTMode: mad(features.map(f => f.dominantRFTMode), medians.dominantRFTMode),
    };
  }
  
  private averageArrays(arrays: number[][]): number[] {
    if (arrays.length === 0) return [];
    const result = new Array(arrays[0].length).fill(0);
    for (const arr of arrays) {
      for (let i = 0; i < arr.length; i++) {
        result[i] += arr[i];
      }
    }
    return result.map(x => x / arrays.length);
  }
  
  // ==========================================================================
  // ANOMALY DETECTION
  // ==========================================================================
  
  /**
   * Score a window against the baseline
   */
  scoreWindow(window: VibrationWindow): AnomalyScore {
    if (!this.baseline) {
      return {
        timestamp: Date.now(),
        overallScore: 0,
        featureScores: {},
        isAnomaly: false,
        consecutiveAnomalies: 0,
        triggeredFeatures: [],
      };
    }
    
    const features = this.extractFeatures(window);
    const scores: Partial<Record<keyof SpectralFeatures, number>> = {};
    const triggered: string[] = [];
    
    // Score each scalar feature
    const scalarFeatures: (keyof SpectralFeatures)[] = [
      'spectralCentroid', 'spectralRolloff', 'spectralEntropy', 
      'spectralFlatness', 'crestFactor', 'kurtosis', 'rmsLevel',
      'rftSparsity', 'rftCoherence', 'dominantRFTMode'
    ];
    
    for (const key of scalarFeatures) {
      const value = features[key] as number;
      const median = this.baseline.featureMedians[key] as number;
      const mad = this.baseline.featureMADs[key] as number;
      
      // MAD-based z-score
      const score = mad > 0 ? Math.abs(value - median) / (1.4826 * mad) : 0;
      scores[key] = score;
      
      if (score > this.config.anomalyThreshold) {
        triggered.push(key);
      }
    }
    
    // Score band energies
    const bandKeys = ['veryLow', 'low', 'mid', 'high'] as const;
    for (const band of bandKeys) {
      const value = features.bandEnergies[band];
      const median = this.baseline.featureMedians.bandEnergies[band];
      const mad = this.baseline.featureMADs.bandEnergies[band];
      
      const score = mad > 0 ? Math.abs(value - median) / (1.4826 * mad) : 0;
      scores[`bandEnergy_${band}` as keyof SpectralFeatures] = score;
      
      if (score > this.config.anomalyThreshold) {
        triggered.push(`bandEnergy_${band}`);
      }
    }
    
    // Overall score: max of individual scores
    const allScores = Object.values(scores).filter(s => typeof s === 'number') as number[];
    const overallScore = Math.max(...allScores, 0);
    
    // Track consecutive anomalies
    const isAnomaly = triggered.length > 0;
    if (isAnomaly) {
      this.consecutiveAnomalies++;
    } else {
      this.consecutiveAnomalies = 0;
    }
    
    // Log event if threshold exceeded
    if (this.consecutiveAnomalies >= this.config.consecutiveWindowsForAlert) {
      const signal = this.preprocessSignal(window.samples);
      const rftCoeffs = this.computeRFT(signal);
      
      this.events.push({
        id: `event_${Date.now()}`,
        timestamp: Date.now(),
        type: 'anomaly_detected',
        score: overallScore,
        features,
        waveformSnapshot: signal.slice(0, 1000), // First 1000 samples
        rftCoefficients: rftCoeffs,
        notes: `Anomaly detected: ${triggered.join(', ')}`,
      });
    }
    
    return {
      timestamp: Date.now(),
      overallScore,
      featureScores: scores,
      isAnomaly,
      consecutiveAnomalies: this.consecutiveAnomalies,
      triggeredFeatures: triggered,
    };
  }
  
  // ==========================================================================
  // EXPORT
  // ==========================================================================
  
  /**
   * Export session data as JSON
   */
  exportSession(): object {
    return {
      config: this.config,
      baseline: this.baseline,
      events: this.events,
      exportedAt: new Date().toISOString(),
      version: '1.0.0',
      platform: 'quantoniumos-mobile',
    };
  }
  
  /**
   * Get recent events
   */
  getEvents(limit: number = 50): SHMEvent[] {
    return this.events.slice(-limit);
  }
  
  /**
   * Get current baseline
   */
  getBaseline(): BaselineModel | null {
    return this.baseline;
  }
  
  /**
   * Load baseline from exported data
   */
  loadBaseline(data: BaselineModel): void {
    this.baseline = data;
  }
}

export default VibrationAnalyzer;
