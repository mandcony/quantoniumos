/**
 * SPDX-License-Identifier: LicenseRef-QuantoniumOS-Claims-NC
 * Copyright (C) 2025 Luis M. Minier / quantoniumos
 * 
 * Structural Health Monitoring Engine
 * 
 * Production-grade vibration analysis for:
 * - Seismic monitoring (earthquake detection, intensity measurement)
 * - Bridge monitoring (modal tracking, deck oscillation, cable tension)
 * - Building sway (wind-induced motion, occupant comfort)
 * - Construction (pile driving, ground-borne vibration compliance)
 * - Industrial (machine health, bearing failure, pump cavitation)
 * 
 * References:
 * - ISO 4866: Mechanical vibration - Guidelines for measurement
 * - DIN 4150-3: Structural vibration in buildings
 * - BS 7385-2: Evaluation of building vibrations
 * - Eurocode 8: Design for earthquake resistance
 */

import { Complex, abs, multiply, add, conj } from '../rft/Complex';

// =============================================================================
// CONSTANTS - Engineering Standards
// =============================================================================

const PHI = (1 + Math.sqrt(5)) / 2;
const GRAVITY = 9.81; // m/s²

// DIN 4150-3 PPV limits (mm/s) for building damage assessment
const DIN_4150_LIMITS = {
  residential: { shortTerm: 5, longTerm: 2.5 },
  commercial: { shortTerm: 15, longTerm: 5 },
  industrial: { shortTerm: 20, longTerm: 10 },
  historic: { shortTerm: 3, longTerm: 1.5 },
};

// ISO 2631 - Human perception thresholds (m/s²)
const HUMAN_PERCEPTION = {
  imperceptible: 0.01,
  barelyPerceptible: 0.04,
  perceptible: 0.125,
  annoying: 0.4,
  uncomfortable: 1.25,
  harmful: 4.0,
};

// Typical structural natural frequencies (Hz)
const TYPICAL_FREQUENCIES = {
  tallBuilding: { min: 0.1, max: 0.5 },      // 50+ story
  midRiseBuilding: { min: 0.5, max: 2.0 },   // 10-30 story
  lowRiseBuilding: { min: 2.0, max: 5.0 },   // 1-5 story
  bridgeDeck: { min: 0.5, max: 3.0 },
  bridgeCable: { min: 0.5, max: 10.0 },
  machinery: { min: 10, max: 1000 },
};

// Earthquake intensity scales
const MERCALLI_INTENSITY = [
  { level: 'I', pga: 0.0017, description: 'Not felt' },
  { level: 'II', pga: 0.014, description: 'Weak' },
  { level: 'III', pga: 0.039, description: 'Slight' },
  { level: 'IV', pga: 0.092, description: 'Moderate' },
  { level: 'V', pga: 0.18, description: 'Rather strong' },
  { level: 'VI', pga: 0.34, description: 'Strong' },
  { level: 'VII', pga: 0.65, description: 'Very strong' },
  { level: 'VIII', pga: 1.24, description: 'Severe' },
  { level: 'IX', pga: 2.0, description: 'Violent' },
  { level: 'X', pga: 999, description: 'Extreme' },
];

// =============================================================================
// TYPES
// =============================================================================

export type SHMDomain = 
  | 'seismic'      // Earthquake detection & monitoring
  | 'bridge'       // Bridge structural health
  | 'building'     // Building sway & occupant comfort
  | 'construction' // Construction vibration compliance
  | 'industrial';  // Machine health monitoring

export interface AccelerometerSample {
  timestamp: number;
  x: number;
  y: number;
  z: number;
  magnitude: number;
}

export interface ModalParameters {
  frequency: number;      // Hz
  dampingRatio: number;   // ζ (typically 0.01-0.10)
  amplitude: number;      // Modal amplitude
  phase: number;          // Phase angle (rad)
  confidence: number;     // 0-1 confidence in detection
}

export interface SeismicEvent {
  timestamp: number;
  duration: number;       // seconds
  pga: number;            // Peak Ground Acceleration (g)
  pgv: number;            // Peak Ground Velocity (m/s)
  pgd: number;            // Peak Ground Displacement (m)
  arias: number;          // Arias Intensity (m/s)
  cav: number;            // Cumulative Absolute Velocity (m/s)
  mercalli: string;       // Modified Mercalli Intensity
  magnitudeEstimate: number;  // Local magnitude estimate
  dominantFrequency: number;  // Hz
  significantDuration: number; // 5-95% Arias duration
}

export interface BridgeHealth {
  timestamp: number;
  deckFrequency: number;       // First vertical mode (Hz)
  torsionalFrequency: number;  // First torsional mode (Hz)
  dampingRatio: number;        // %
  frequencyShift: number;      // % change from baseline
  deckAcceleration: number;    // m/s² RMS
  trafficIndex: number;        // 0-1 traffic intensity
  windResponse: number;        // Wind-induced vibration level
  cableForceIndex: number;     // Relative cable tension indicator
  healthIndex: number;         // 0-100 overall health score
  alerts: string[];
}

export interface BuildingResponse {
  timestamp: number;
  swayFrequency: number;       // Fundamental frequency (Hz)
  swayAmplitude: number;       // mm displacement estimate
  swayDirection: number;       // degrees from North
  dampingRatio: number;        // %
  accelerationRMS: number;     // m/s²
  peakAcceleration: number;    // m/s²
  comfortLevel: string;        // ISO 10137 assessment
  windSpeed: number;           // Estimated from response (m/s)
  torsionIndex: number;        // Torsional/translational ratio
  humanPerception: string;     // Perceptibility assessment
  alerts: string[];
}

export interface ConstructionCompliance {
  timestamp: number;
  ppv: number;                 // Peak Particle Velocity (mm/s)
  frequency: number;           // Dominant frequency (Hz)
  din4150Category: string;     // Building category
  complianceStatus: string;    // 'compliant' | 'warning' | 'violation'
  margin: number;              // % below limit
  crestFactor: number;         // Peak/RMS ratio
  vibrationDose: number;       // VDV (m/s^1.75)
  sourceType: string;          // Estimated source type
  duration: number;            // Event duration (s)
  alerts: string[];
}

export interface MachineHealth {
  timestamp: number;
  rpm: number;                 // Estimated rotation speed
  fundamentalFrequency: number;// 1X frequency (Hz)
  harmonics: number[];         // Amplitudes at 2X, 3X, 4X...
  overallVibration: number;    // mm/s RMS (ISO 10816)
  unbalanceIndex: number;      // 1X amplitude indicator
  misalignmentIndex: number;   // 2X amplitude indicator
  bearingCondition: string;    // 'good' | 'acceptable' | 'unsatisfactory' | 'unacceptable'
  loosenesIndex: number;       // Sub-harmonic indicator
  cavitationIndex: number;     // High-freq noise indicator
  healthScore: number;         // 0-100
  remainingLife: string;       // Estimated remaining life
  alerts: string[];
}

export interface SpectralAnalysis {
  frequencies: number[];
  magnitudes: number[];
  phases: number[];
  psd: number[];              // Power Spectral Density
  rmsSpectrum: number[];
  peakFrequencies: number[];
  peakAmplitudes: number[];
  bandEnergies: Record<string, number>;
  spectralCentroid: number;
  spectralRolloff: number;
  spectralEntropy: number;
  rftSparsity: number;
  rftCoherence: number;
}

export interface SHMResult {
  domain: SHMDomain;
  timestamp: number;
  rawData: AccelerometerSample[];
  spectral: SpectralAnalysis;
  modes: ModalParameters[];
  seismic?: SeismicEvent;
  bridge?: BridgeHealth;
  building?: BuildingResponse;
  construction?: ConstructionCompliance;
  machine?: MachineHealth;
  alerts: string[];
  confidence: number;
}

// =============================================================================
// SHM ENGINE CLASS
// =============================================================================

export class SHMEngine {
  private domain: SHMDomain;
  private sampleRate: number;
  private baseline: SpectralAnalysis | null = null;
  private baselineFrequencies: number[] = [];
  private history: SHMResult[] = [];
  
  constructor(domain: SHMDomain = 'building', sampleRate: number = 100) {
    this.domain = domain;
    this.sampleRate = sampleRate;
  }
  
  // ===========================================================================
  // PUBLIC API
  // ===========================================================================
  
  /**
   * Main analysis entry point - processes raw accelerometer data
   */
  analyze(samples: AccelerometerSample[]): SHMResult {
    if (samples.length < 32) {
      throw new Error('Minimum 32 samples required for analysis');
    }
    
    // Extract signal components
    const signal = samples.map(s => s.magnitude);
    const xSignal = samples.map(s => s.x);
    const ySignal = samples.map(s => s.y);
    const zSignal = samples.map(s => s.z);
    
    // Core spectral analysis
    const spectral = this.computeSpectralAnalysis(signal);
    
    // Modal parameter extraction
    const modes = this.extractModes(spectral);
    
    // Domain-specific analysis
    const result: SHMResult = {
      domain: this.domain,
      timestamp: samples[samples.length - 1].timestamp,
      rawData: samples,
      spectral,
      modes,
      alerts: [],
      confidence: this.computeConfidence(spectral, modes),
    };
    
    switch (this.domain) {
      case 'seismic':
        result.seismic = this.analyzeSeismic(samples, spectral);
        result.alerts = this.getSeismicAlerts(result.seismic);
        break;
      case 'bridge':
        result.bridge = this.analyzeBridge(samples, spectral, modes);
        result.alerts = result.bridge.alerts;
        break;
      case 'building':
        result.building = this.analyzeBuilding(samples, xSignal, ySignal, spectral, modes);
        result.alerts = result.building.alerts;
        break;
      case 'construction':
        result.construction = this.analyzeConstruction(samples, spectral);
        result.alerts = result.construction.alerts;
        break;
      case 'industrial':
        result.machine = this.analyzeMachine(samples, spectral);
        result.alerts = result.machine.alerts;
        break;
    }
    
    this.history.push(result);
    if (this.history.length > 1000) this.history.shift();
    
    return result;
  }
  
  /**
   * Set baseline for comparison
   */
  setBaseline(samples: AccelerometerSample[]): void {
    const signal = samples.map(s => s.magnitude);
    this.baseline = this.computeSpectralAnalysis(signal);
    this.baselineFrequencies = this.baseline.peakFrequencies.slice(0, 5);
  }
  
  /**
   * Change monitoring domain
   */
  setDomain(domain: SHMDomain): void {
    this.domain = domain;
  }
  
  /**
   * Get historical trend
   */
  getTrend(metric: string, windowSize: number = 20): number[] {
    return this.history.slice(-windowSize).map(r => {
      switch (metric) {
        case 'frequency':
          return r.modes[0]?.frequency ?? 0;
        case 'amplitude':
          return r.spectral.peakAmplitudes[0] ?? 0;
        case 'rms':
          return this.computeRMS(r.rawData.map(s => s.magnitude));
        default:
          return 0;
      }
    });
  }
  
  // ===========================================================================
  // SPECTRAL ANALYSIS
  // ===========================================================================
  
  private computeSpectralAnalysis(signal: number[]): SpectralAnalysis {
    // Preprocessing
    const processed = this.preprocess(signal);
    
    // FFT computation
    const fft = this.computeFFT(processed);
    const N = fft.length;
    const freqResolution = this.sampleRate / N;
    
    // Frequency axis
    const frequencies: number[] = [];
    for (let i = 0; i < N / 2; i++) {
      frequencies.push(i * freqResolution);
    }
    
    // Magnitude and phase
    const magnitudes: number[] = [];
    const phases: number[] = [];
    for (let i = 0; i < N / 2; i++) {
      magnitudes.push(abs(fft[i]) * 2 / N);
      phases.push(Math.atan2(fft[i].im, fft[i].re));
    }
    
    // Power Spectral Density (Welch method approximation)
    const psd = magnitudes.map(m => m * m / freqResolution);
    
    // RMS spectrum
    const rmsSpectrum = magnitudes.map(m => m / Math.sqrt(2));
    
    // Peak detection
    const peaks = this.findPeaks(magnitudes, frequencies);
    
    // Band energies
    const bandEnergies = this.computeBandEnergies(frequencies, psd);
    
    // Spectral features
    const spectralCentroid = this.computeSpectralCentroid(frequencies, magnitudes);
    const spectralRolloff = this.computeSpectralRolloff(frequencies, magnitudes, 0.85);
    const spectralEntropy = this.computeSpectralEntropy(magnitudes);
    
    // RFT analysis
    const rftCoeffs = this.computeRFT(processed);
    const rftSparsity = this.computeSparsity(rftCoeffs);
    const rftCoherence = this.baseline 
      ? this.computeCoherence(magnitudes, this.baseline.magnitudes)
      : 1.0;
    
    return {
      frequencies,
      magnitudes,
      phases,
      psd,
      rmsSpectrum,
      peakFrequencies: peaks.map(p => p.frequency),
      peakAmplitudes: peaks.map(p => p.amplitude),
      bandEnergies,
      spectralCentroid,
      spectralRolloff,
      spectralEntropy,
      rftSparsity,
      rftCoherence,
    };
  }
  
  // ===========================================================================
  // MODAL ANALYSIS
  // ===========================================================================
  
  private extractModes(spectral: SpectralAnalysis): ModalParameters[] {
    const modes: ModalParameters[] = [];
    
    for (let i = 0; i < Math.min(5, spectral.peakFrequencies.length); i++) {
      const freq = spectral.peakFrequencies[i];
      const amp = spectral.peakAmplitudes[i];
      
      // Find index in frequency array
      const idx = spectral.frequencies.findIndex(f => Math.abs(f - freq) < 0.1);
      const phase = idx >= 0 ? spectral.phases[idx] : 0;
      
      // Estimate damping using half-power bandwidth
      const damping = this.estimateDamping(spectral, freq);
      
      // Confidence based on peak prominence
      const totalEnergy = spectral.magnitudes.reduce((a, b) => a + b * b, 0);
      const confidence = Math.min(1, (amp * amp) / (totalEnergy + 1e-10) * 10);
      
      modes.push({
        frequency: freq,
        dampingRatio: damping,
        amplitude: amp,
        phase,
        confidence,
      });
    }
    
    return modes;
  }
  
  private estimateDamping(spectral: SpectralAnalysis, peakFreq: number): number {
    const idx = spectral.frequencies.findIndex(f => Math.abs(f - peakFreq) < 0.1);
    if (idx < 0) return 0.02; // Default 2% damping
    
    const peakMag = spectral.magnitudes[idx];
    const halfPower = peakMag / Math.sqrt(2);
    
    // Find half-power points
    let f1 = peakFreq, f2 = peakFreq;
    for (let i = idx; i >= 0; i--) {
      if (spectral.magnitudes[i] <= halfPower) {
        f1 = spectral.frequencies[i];
        break;
      }
    }
    for (let i = idx; i < spectral.magnitudes.length; i++) {
      if (spectral.magnitudes[i] <= halfPower) {
        f2 = spectral.frequencies[i];
        break;
      }
    }
    
    // Damping ratio from bandwidth
    const bandwidth = f2 - f1;
    const damping = bandwidth / (2 * peakFreq);
    
    return Math.max(0.001, Math.min(0.2, damping)); // Clamp to realistic range
  }
  
  // ===========================================================================
  // SEISMIC ANALYSIS
  // ===========================================================================
  
  private analyzeSeismic(
    samples: AccelerometerSample[], 
    spectral: SpectralAnalysis
  ): SeismicEvent {
    const signal = samples.map(s => s.magnitude);
    const dt = 1 / this.sampleRate;
    const duration = samples.length * dt;
    
    // Peak Ground Acceleration (g)
    const pga = Math.max(...signal.map(s => Math.abs(s - 1))) * GRAVITY;
    
    // Peak Ground Velocity (integrate acceleration)
    const velocity = this.integrate(signal.map(s => (s - 1) * GRAVITY), dt);
    const pgv = Math.max(...velocity.map(Math.abs));
    
    // Peak Ground Displacement (integrate velocity)
    const displacement = this.integrate(velocity, dt);
    const pgd = Math.max(...displacement.map(Math.abs));
    
    // Arias Intensity
    const arias = (Math.PI / (2 * GRAVITY)) * 
      signal.reduce((sum, a) => sum + ((a - 1) * GRAVITY) ** 2 * dt, 0);
    
    // Cumulative Absolute Velocity
    const cav = signal.reduce((sum, a) => sum + Math.abs((a - 1) * GRAVITY) * dt, 0);
    
    // Modified Mercalli Intensity
    const pgaG = pga / GRAVITY;
    const mercalli = MERCALLI_INTENSITY.find(m => pgaG < m.pga)?.level ?? 'X';
    
    // Magnitude estimate (simplified Gutenberg-Richter)
    // log(A) = M - logR - 1.73 (assuming R=10km)
    const magnitudeEstimate = Math.log10(pga * 1000) + 2.73;
    
    // Dominant frequency
    const dominantFrequency = spectral.peakFrequencies[0] ?? 1;
    
    // Significant duration (5-95% Arias intensity)
    const ariasTimeSeries: number[] = [];
    let cumArias = 0;
    for (const a of signal) {
      cumArias += (Math.PI / (2 * GRAVITY)) * ((a - 1) * GRAVITY) ** 2 * dt;
      ariasTimeSeries.push(cumArias);
    }
    const t5 = ariasTimeSeries.findIndex(a => a >= 0.05 * arias) * dt;
    const t95 = ariasTimeSeries.findIndex(a => a >= 0.95 * arias) * dt;
    const significantDuration = Math.max(0, t95 - t5);
    
    return {
      timestamp: samples[samples.length - 1].timestamp,
      duration,
      pga: pga / GRAVITY, // Report in g
      pgv,
      pgd,
      arias,
      cav,
      mercalli,
      magnitudeEstimate: Math.max(0, magnitudeEstimate),
      dominantFrequency,
      significantDuration,
    };
  }
  
  private getSeismicAlerts(seismic: SeismicEvent): string[] {
    const alerts: string[] = [];
    
    if (seismic.pga > 0.1) {
      alerts.push(`⚠️ Strong shaking detected: ${seismic.pga.toFixed(3)}g`);
    }
    if (seismic.pga > 0.3) {
      alerts.push(`🚨 SEVERE SHAKING: ${seismic.pga.toFixed(3)}g - Take cover!`);
    }
    if (seismic.mercalli !== 'I' && seismic.mercalli !== 'II') {
      alerts.push(`Intensity: ${seismic.mercalli} (${MERCALLI_INTENSITY.find(m => m.level === seismic.mercalli)?.description})`);
    }
    if (seismic.cav > 0.16) {
      alerts.push(`⚠️ High CAV: potential structural damage threshold`);
    }
    
    return alerts;
  }
  
  // ===========================================================================
  // BRIDGE ANALYSIS
  // ===========================================================================
  
  private analyzeBridge(
    samples: AccelerometerSample[],
    spectral: SpectralAnalysis,
    modes: ModalParameters[]
  ): BridgeHealth {
    const alerts: string[] = [];
    const signal = samples.map(s => s.magnitude);
    
    // Extract deck vertical frequency (typically first mode)
    const deckFrequency = modes[0]?.frequency ?? 0;
    
    // Torsional frequency (typically higher mode)
    const torsionalFrequency = modes.length > 1 ? modes[1].frequency : 0;
    
    // Damping ratio
    const dampingRatio = modes[0]?.dampingRatio ?? 0.02;
    
    // Frequency shift from baseline
    let frequencyShift = 0;
    if (this.baselineFrequencies.length > 0) {
      const baselineFreq = this.baselineFrequencies[0];
      frequencyShift = ((deckFrequency - baselineFreq) / baselineFreq) * 100;
    }
    
    // RMS acceleration
    const deckAcceleration = this.computeRMS(signal.map(s => (s - 1) * GRAVITY));
    
    // Traffic index (based on vibration level and frequency content)
    const trafficIndex = Math.min(1, deckAcceleration / 0.5);
    
    // Wind response (low frequency energy)
    const lowFreqEnergy = spectral.bandEnergies['0.1-2Hz'] ?? 0;
    const totalEnergy = Object.values(spectral.bandEnergies).reduce((a, b) => a + b, 0);
    const windResponse = totalEnergy > 0 ? lowFreqEnergy / totalEnergy : 0;
    
    // Cable force index (frequency ratio indicator)
    const cableForceIndex = torsionalFrequency > 0 ? deckFrequency / torsionalFrequency : 1;
    
    // Overall health index
    let healthIndex = 100;
    
    // Frequency shift penalty
    if (Math.abs(frequencyShift) > 5) {
      healthIndex -= Math.abs(frequencyShift) * 2;
      alerts.push(`⚠️ Natural frequency shifted by ${frequencyShift.toFixed(1)}%`);
    }
    if (Math.abs(frequencyShift) > 15) {
      alerts.push(`🚨 CRITICAL: Large frequency shift indicates possible damage`);
    }
    
    // Damping anomaly
    if (dampingRatio > 0.1) {
      healthIndex -= 10;
      alerts.push(`⚠️ High damping ratio: ${(dampingRatio * 100).toFixed(1)}%`);
    }
    
    // Excessive vibration
    if (deckAcceleration > 1.0) {
      healthIndex -= 20;
      alerts.push(`⚠️ High deck acceleration: ${deckAcceleration.toFixed(2)} m/s²`);
    }
    
    // Coherence check
    if (spectral.rftCoherence < 0.8 && this.baseline) {
      healthIndex -= 15;
      alerts.push(`⚠️ Low spectral coherence: possible structural change`);
    }
    
    return {
      timestamp: samples[samples.length - 1].timestamp,
      deckFrequency,
      torsionalFrequency,
      dampingRatio: dampingRatio * 100,
      frequencyShift,
      deckAcceleration,
      trafficIndex,
      windResponse,
      cableForceIndex,
      healthIndex: Math.max(0, healthIndex),
      alerts,
    };
  }
  
  // ===========================================================================
  // BUILDING ANALYSIS
  // ===========================================================================
  
  private analyzeBuilding(
    samples: AccelerometerSample[],
    xSignal: number[],
    ySignal: number[],
    spectral: SpectralAnalysis,
    modes: ModalParameters[]
  ): BuildingResponse {
    const alerts: string[] = [];
    const signal = samples.map(s => s.magnitude);
    
    // Fundamental sway frequency
    const swayFrequency = modes[0]?.frequency ?? 0;
    
    // Estimate displacement from acceleration (double integration)
    const dt = 1 / this.sampleRate;
    const velocity = this.integrate(signal.map(s => (s - 1) * GRAVITY), dt);
    const displacement = this.integrate(velocity, dt);
    const swayAmplitude = Math.max(...displacement.map(Math.abs)) * 1000; // mm
    
    // Sway direction from X/Y components
    const xRMS = this.computeRMS(xSignal);
    const yRMS = this.computeRMS(ySignal);
    const swayDirection = Math.atan2(yRMS, xRMS) * (180 / Math.PI);
    
    // Damping ratio
    const dampingRatio = modes[0]?.dampingRatio ?? 0.02;
    
    // Acceleration metrics
    const accelerationRMS = this.computeRMS(signal.map(s => Math.abs(s - 1) * GRAVITY));
    const peakAcceleration = Math.max(...signal.map(s => Math.abs(s - 1) * GRAVITY));
    
    // Comfort level (ISO 10137)
    let comfortLevel = 'Comfortable';
    if (accelerationRMS > 0.05) comfortLevel = 'Perceptible';
    if (accelerationRMS > 0.1) comfortLevel = 'Annoying';
    if (accelerationRMS > 0.25) {
      comfortLevel = 'Uncomfortable';
      alerts.push(`⚠️ Occupant discomfort likely`);
    }
    if (accelerationRMS > 0.5) {
      comfortLevel = 'Unacceptable';
      alerts.push(`🚨 Unacceptable vibration level for occupants`);
    }
    
    // Wind speed estimate (simplified)
    // f = 46/H for concrete, V_cr = 6.3 * f * sqrt(B*D/m)
    // Simplified: V ≈ amplitude * frequency * constant
    const windSpeed = swayAmplitude * swayFrequency * 0.1; // Rough estimate
    
    // Torsion index (ratio of torsional to translational motion)
    const highFreqEnergy = spectral.bandEnergies['2-10Hz'] ?? 0;
    const lowFreqEnergy = spectral.bandEnergies['0.1-2Hz'] ?? 0;
    const torsionIndex = highFreqEnergy / (lowFreqEnergy + 1e-10);
    
    // Human perception
    let humanPerception = 'Not perceptible';
    if (accelerationRMS > HUMAN_PERCEPTION.imperceptible) humanPerception = 'Imperceptible threshold';
    if (accelerationRMS > HUMAN_PERCEPTION.barelyPerceptible) humanPerception = 'Barely perceptible';
    if (accelerationRMS > HUMAN_PERCEPTION.perceptible) humanPerception = 'Perceptible';
    if (accelerationRMS > HUMAN_PERCEPTION.annoying) humanPerception = 'Annoying';
    if (accelerationRMS > HUMAN_PERCEPTION.uncomfortable) humanPerception = 'Uncomfortable';
    
    // Frequency check for building type
    if (swayFrequency > 0 && swayFrequency < TYPICAL_FREQUENCIES.tallBuilding.min) {
      alerts.push(`⚠️ Very low frequency (${swayFrequency.toFixed(2)} Hz) - possible sensor issue or very tall building`);
    }
    
    return {
      timestamp: samples[samples.length - 1].timestamp,
      swayFrequency,
      swayAmplitude,
      swayDirection,
      dampingRatio: dampingRatio * 100,
      accelerationRMS,
      peakAcceleration,
      comfortLevel,
      windSpeed,
      torsionIndex,
      humanPerception,
      alerts,
    };
  }
  
  // ===========================================================================
  // CONSTRUCTION COMPLIANCE
  // ===========================================================================
  
  private analyzeConstruction(
    samples: AccelerometerSample[],
    spectral: SpectralAnalysis
  ): ConstructionCompliance {
    const alerts: string[] = [];
    const signal = samples.map(s => s.magnitude);
    const dt = 1 / this.sampleRate;
    
    // Peak Particle Velocity (PPV) - integrate acceleration
    const velocity = this.integrate(signal.map(s => (s - 1) * GRAVITY), dt);
    const ppv = Math.max(...velocity.map(Math.abs)) * 1000; // mm/s
    
    // Dominant frequency
    const frequency = spectral.peakFrequencies[0] ?? 10;
    
    // Building category (default to residential for safety)
    const category = 'residential';
    const limit = DIN_4150_LIMITS[category].shortTerm;
    
    // Compliance status
    let complianceStatus = 'compliant';
    let margin = ((limit - ppv) / limit) * 100;
    
    if (ppv > limit * 0.8) {
      complianceStatus = 'warning';
      alerts.push(`⚠️ PPV at ${((ppv / limit) * 100).toFixed(0)}% of DIN 4150-3 limit`);
    }
    if (ppv > limit) {
      complianceStatus = 'violation';
      alerts.push(`🚨 DIN 4150-3 VIOLATION: PPV ${ppv.toFixed(1)} mm/s exceeds ${limit} mm/s limit`);
      margin = -((ppv - limit) / limit) * 100;
    }
    
    // Crest factor
    const rms = this.computeRMS(velocity) * 1000;
    const crestFactor = ppv / (rms + 1e-10);
    
    // Vibration Dose Value (VDV)
    const accelerationG = signal.map(s => Math.abs(s - 1) * GRAVITY);
    const vdv = Math.pow(
      accelerationG.reduce((sum, a) => sum + Math.pow(a, 4) * dt, 0),
      0.25
    );
    
    // Source type estimation
    let sourceType = 'Unknown';
    if (frequency < 5 && crestFactor > 5) sourceType = 'Impact pile driving';
    else if (frequency < 10 && crestFactor < 3) sourceType = 'Vibratory compaction';
    else if (frequency > 20 && frequency < 100) sourceType = 'Construction machinery';
    else if (frequency > 100) sourceType = 'High-speed equipment';
    else sourceType = 'General construction';
    
    // Event duration
    const threshold = 0.001;
    const activeIndices = signal.map((s, i) => Math.abs(s - 1) > threshold ? i : -1).filter(i => i >= 0);
    const duration = activeIndices.length > 0 
      ? (activeIndices[activeIndices.length - 1] - activeIndices[0]) * dt
      : 0;
    
    return {
      timestamp: samples[samples.length - 1].timestamp,
      ppv,
      frequency,
      din4150Category: category,
      complianceStatus,
      margin,
      crestFactor,
      vibrationDose: vdv,
      sourceType,
      duration,
      alerts,
    };
  }
  
  // ===========================================================================
  // MACHINE HEALTH
  // ===========================================================================
  
  private analyzeMachine(
    samples: AccelerometerSample[],
    spectral: SpectralAnalysis
  ): MachineHealth {
    const alerts: string[] = [];
    const signal = samples.map(s => s.magnitude);
    
    // Estimate RPM from dominant frequency
    const fundamentalFrequency = spectral.peakFrequencies[0] ?? 0;
    const rpm = fundamentalFrequency * 60;
    
    // Extract harmonics (2X, 3X, 4X...)
    const harmonics: number[] = [];
    for (let h = 2; h <= 5; h++) {
      const targetFreq = fundamentalFrequency * h;
      const idx = spectral.frequencies.findIndex(f => Math.abs(f - targetFreq) < fundamentalFrequency * 0.1);
      harmonics.push(idx >= 0 ? spectral.magnitudes[idx] : 0);
    }
    
    // Overall vibration (ISO 10816 velocity)
    const velocity = this.integrate(signal.map(s => (s - 1) * GRAVITY), 1 / this.sampleRate);
    const overallVibration = this.computeRMS(velocity) * 1000; // mm/s
    
    // Fault indicators
    const unbalanceIndex = spectral.peakAmplitudes[0] ?? 0;
    const misalignmentIndex = harmonics[0] ?? 0;
    const loosenesIndex = this.computeSubharmonicEnergy(spectral, fundamentalFrequency);
    const cavitationIndex = spectral.bandEnergies['high'] ?? 0;
    
    // Bearing condition (ISO 10816 Class I limits)
    let bearingCondition = 'good';
    if (overallVibration > 0.71) bearingCondition = 'acceptable';
    if (overallVibration > 1.8) {
      bearingCondition = 'unsatisfactory';
      alerts.push(`⚠️ High vibration: ${overallVibration.toFixed(2)} mm/s RMS`);
    }
    if (overallVibration > 4.5) {
      bearingCondition = 'unacceptable';
      alerts.push(`🚨 CRITICAL: Vibration ${overallVibration.toFixed(2)} mm/s exceeds ISO 10816 limits`);
    }
    
    // Fault detection alerts
    if (misalignmentIndex > unbalanceIndex * 0.5) {
      alerts.push(`⚠️ Possible misalignment (high 2X component)`);
    }
    if (loosenesIndex > 0.1) {
      alerts.push(`⚠️ Possible mechanical looseness (sub-harmonics detected)`);
    }
    if (cavitationIndex > spectral.bandEnergies['mid'] * 0.5) {
      alerts.push(`⚠️ High-frequency noise - check for cavitation or bearing wear`);
    }
    
    // Health score
    let healthScore = 100;
    healthScore -= Math.min(30, overallVibration * 10);
    healthScore -= Math.min(20, misalignmentIndex / unbalanceIndex * 10);
    healthScore -= Math.min(20, loosenesIndex * 100);
    healthScore = Math.max(0, healthScore);
    
    // Remaining life estimate
    let remainingLife = 'Normal';
    if (healthScore < 80) remainingLife = 'Monitor closely';
    if (healthScore < 60) remainingLife = 'Schedule maintenance';
    if (healthScore < 40) remainingLife = 'Urgent maintenance required';
    if (healthScore < 20) remainingLife = 'Immediate action needed';
    
    return {
      timestamp: samples[samples.length - 1].timestamp,
      rpm,
      fundamentalFrequency,
      harmonics,
      overallVibration,
      unbalanceIndex,
      misalignmentIndex,
      bearingCondition,
      loosenesIndex,
      cavitationIndex,
      healthScore,
      remainingLife,
      alerts,
    };
  }
  
  private computeSubharmonicEnergy(spectral: SpectralAnalysis, fundamentalFreq: number): number {
    // Energy below fundamental frequency
    let subEnergy = 0;
    let totalEnergy = 0;
    
    for (let i = 0; i < spectral.frequencies.length; i++) {
      const energy = spectral.magnitudes[i] ** 2;
      totalEnergy += energy;
      if (spectral.frequencies[i] < fundamentalFreq * 0.9) {
        subEnergy += energy;
      }
    }
    
    return totalEnergy > 0 ? subEnergy / totalEnergy : 0;
  }
  
  // ===========================================================================
  // SIGNAL PROCESSING UTILITIES
  // ===========================================================================
  
  private preprocess(signal: number[]): number[] {
    // Remove DC offset
    const mean = signal.reduce((a, b) => a + b, 0) / signal.length;
    let processed = signal.map(x => x - mean);
    
    // Apply Hanning window
    const N = processed.length;
    processed = processed.map((x, i) => x * (0.5 - 0.5 * Math.cos(2 * Math.PI * i / N)));
    
    return processed;
  }
  
  private computeFFT(signal: number[]): Complex[] {
    const N = signal.length;
    const size = Math.pow(2, Math.ceil(Math.log2(N)));
    
    // Pad to power of 2
    const padded = [...signal];
    while (padded.length < size) padded.push(0);
    
    // Bit-reversal permutation
    const output: Complex[] = padded.map(x => ({ re: x, im: 0 }));
    const bits = Math.log2(size);
    
    for (let i = 0; i < size; i++) {
      const j = this.reverseBits(i, bits);
      if (j > i) [output[i], output[j]] = [output[j], output[i]];
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
          output[i + j + len / 2] = { re: u.re - v.re, im: u.im - v.im };
          
          w = multiply(w, wlen);
        }
      }
    }
    
    return output;
  }
  
  private computeRFT(signal: number[]): Complex[] {
    // Simplified Φ-RFT using golden-ratio phase modulation
    const N = signal.length;
    const output: Complex[] = [];
    
    for (let k = 0; k < N; k++) {
      let sum: Complex = { re: 0, im: 0 };
      
      for (let n = 0; n < N; n++) {
        // Golden-ratio phase: frac(k*n/φ) * 2π
        const phiPhase = ((k * n / PHI) % 1) * 2 * Math.PI;
        // Standard DFT phase
        const dftPhase = -2 * Math.PI * k * n / N;
        // Combined phase with golden-ratio weighting
        const phase = dftPhase + phiPhase * 0.1; // Subtle Φ modulation
        
        sum.re += signal[n] * Math.cos(phase);
        sum.im += signal[n] * Math.sin(phase);
      }
      
      output.push({ re: sum.re / N, im: sum.im / N });
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
  
  private integrate(signal: number[], dt: number): number[] {
    const result: number[] = [0];
    for (let i = 1; i < signal.length; i++) {
      result.push(result[i - 1] + signal[i] * dt);
    }
    // Remove drift (high-pass filter)
    const mean = result.reduce((a, b) => a + b, 0) / result.length;
    return result.map(x => x - mean);
  }
  
  private computeRMS(signal: number[]): number {
    const sumSquares = signal.reduce((sum, x) => sum + x * x, 0);
    return Math.sqrt(sumSquares / signal.length);
  }
  
  private findPeaks(
    magnitudes: number[], 
    frequencies: number[]
  ): Array<{ frequency: number; amplitude: number }> {
    const peaks: Array<{ frequency: number; amplitude: number; index: number }> = [];
    
    for (let i = 2; i < magnitudes.length - 2; i++) {
      if (
        magnitudes[i] > magnitudes[i - 1] &&
        magnitudes[i] > magnitudes[i - 2] &&
        magnitudes[i] > magnitudes[i + 1] &&
        magnitudes[i] > magnitudes[i + 2]
      ) {
        peaks.push({
          frequency: frequencies[i],
          amplitude: magnitudes[i],
          index: i,
        });
      }
    }
    
    // Sort by amplitude and return top peaks
    return peaks
      .sort((a, b) => b.amplitude - a.amplitude)
      .slice(0, 10)
      .map(p => ({ frequency: p.frequency, amplitude: p.amplitude }));
  }
  
  private computeBandEnergies(
    frequencies: number[], 
    psd: number[]
  ): Record<string, number> {
    const bands: Record<string, { min: number; max: number }> = {
      '0.1-2Hz': { min: 0.1, max: 2 },
      '2-10Hz': { min: 2, max: 10 },
      '10-50Hz': { min: 10, max: 50 },
      '50-200Hz': { min: 50, max: 200 },
      'veryLow': { min: 0.5, max: 2 },
      'low': { min: 2, max: 10 },
      'mid': { min: 10, max: 30 },
      'high': { min: 30, max: 100 },
    };
    
    const energies: Record<string, number> = {};
    
    for (const [name, range] of Object.entries(bands)) {
      let energy = 0;
      for (let i = 0; i < frequencies.length; i++) {
        if (frequencies[i] >= range.min && frequencies[i] < range.max) {
          energy += psd[i];
        }
      }
      energies[name] = energy;
    }
    
    return energies;
  }
  
  private computeSpectralCentroid(frequencies: number[], magnitudes: number[]): number {
    let weightedSum = 0;
    let totalMag = 0;
    
    for (let i = 0; i < frequencies.length; i++) {
      weightedSum += frequencies[i] * magnitudes[i];
      totalMag += magnitudes[i];
    }
    
    return totalMag > 0 ? weightedSum / totalMag : 0;
  }
  
  private computeSpectralRolloff(
    frequencies: number[], 
    magnitudes: number[], 
    threshold: number
  ): number {
    const totalEnergy = magnitudes.reduce((a, b) => a + b, 0);
    const targetEnergy = totalEnergy * threshold;
    
    let cumulative = 0;
    for (let i = 0; i < magnitudes.length; i++) {
      cumulative += magnitudes[i];
      if (cumulative >= targetEnergy) {
        return frequencies[i];
      }
    }
    
    return frequencies[frequencies.length - 1];
  }
  
  private computeSpectralEntropy(magnitudes: number[]): number {
    const total = magnitudes.reduce((a, b) => a + b, 0);
    if (total === 0) return 0;
    
    const normalized = magnitudes.map(m => m / total);
    let entropy = 0;
    
    for (const p of normalized) {
      if (p > 0) {
        entropy -= p * Math.log2(p);
      }
    }
    
    // Normalize by max entropy
    const maxEntropy = Math.log2(magnitudes.length);
    return maxEntropy > 0 ? entropy / maxEntropy : 0;
  }
  
  private computeSparsity(coefficients: Complex[]): number {
    const magnitudes = coefficients.map(c => abs(c) ** 2);
    const totalEnergy = magnitudes.reduce((a, b) => a + b, 0);
    const sorted = [...magnitudes].sort((a, b) => b - a);
    
    let cumulative = 0;
    let count = 0;
    
    for (const mag of sorted) {
      cumulative += mag;
      count++;
      if (cumulative >= 0.99 * totalEnergy) break;
    }
    
    return count / coefficients.length;
  }
  
  private computeCoherence(current: number[], baseline: number[]): number {
    if (current.length !== baseline.length) return 0;
    
    let dotProduct = 0;
    let normCurrent = 0;
    let normBaseline = 0;
    
    for (let i = 0; i < current.length; i++) {
      dotProduct += current[i] * baseline[i];
      normCurrent += current[i] ** 2;
      normBaseline += baseline[i] ** 2;
    }
    
    const denom = Math.sqrt(normCurrent * normBaseline);
    return denom > 0 ? dotProduct / denom : 0;
  }
  
  private computeConfidence(spectral: SpectralAnalysis, modes: ModalParameters[]): number {
    // Base confidence on signal quality indicators
    let confidence = 1.0;
    
    // Low entropy = more tonal = higher confidence
    confidence *= (1 - spectral.spectralEntropy * 0.5);
    
    // Strong peaks = higher confidence
    if (spectral.peakAmplitudes.length > 0) {
      const peakStrength = spectral.peakAmplitudes[0] / 
        (spectral.magnitudes.reduce((a, b) => a + b, 0) / spectral.magnitudes.length + 1e-10);
      confidence *= Math.min(1, peakStrength / 10);
    }
    
    // Mode confidence
    if (modes.length > 0) {
      confidence *= modes[0].confidence;
    }
    
    return Math.max(0, Math.min(1, confidence));
  }
}

// =============================================================================
// EXPORTS
// =============================================================================

export {
  DIN_4150_LIMITS,
  HUMAN_PERCEPTION,
  TYPICAL_FREQUENCIES,
  MERCALLI_INTENSITY,
  GRAVITY,
};
