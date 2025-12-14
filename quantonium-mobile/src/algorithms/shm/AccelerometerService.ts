/**
 * SPDX-License-Identifier: LicenseRef-QuantoniumOS-Claims-NC
 * Copyright (C) 2025 Luis M. Minier / quantoniumos
 * 
 * Accelerometer Service - iOS/Android sensor interface
 * 
 * Uses Expo Sensors API to capture accelerometer data
 */

import { Accelerometer, AccelerometerMeasurement } from 'expo-sensors';
import { AccelerometerSample, VibrationWindow } from './VibrationAnalyzer';

export interface AccelerometerConfig {
  updateIntervalMs: number;  // Target sample interval (actual may vary)
  useGravity: boolean;       // Include gravity or use linear acceleration
}

export const DEFAULT_ACCELEROMETER_CONFIG: AccelerometerConfig = {
  updateIntervalMs: 10,      // ~100 Hz target
  useGravity: false,
};

export type AccelerometerCallback = (sample: AccelerometerSample) => void;
export type WindowCallback = (window: VibrationWindow) => void;

export class AccelerometerService {
  private config: AccelerometerConfig;
  private subscription: ReturnType<typeof Accelerometer.addListener> | null = null;
  private isRunning: boolean = false;
  
  // Buffering
  private sampleBuffer: AccelerometerSample[] = [];
  private windowDurationMs: number = 10000;
  private windowOverlap: number = 0.5;
  private lastWindowEndTime: number = 0;
  private windowCounter: number = 0;
  
  // Callbacks
  private sampleCallbacks: AccelerometerCallback[] = [];
  private windowCallbacks: WindowCallback[] = [];
  
  constructor(config: Partial<AccelerometerConfig> = {}) {
    this.config = { ...DEFAULT_ACCELEROMETER_CONFIG, ...config };
  }
  
  /**
   * Check if accelerometer is available
   */
  async isAvailable(): Promise<boolean> {
    try {
      const { status } = await Accelerometer.getPermissionsAsync();
      if (status !== 'granted') {
        const { status: newStatus } = await Accelerometer.requestPermissionsAsync();
        return newStatus === 'granted';
      }
      return true;
    } catch (e) {
      console.warn('Accelerometer permission check failed:', e);
      return false;
    }
  }
  
  /**
   * Start capturing accelerometer data
   */
  async start(windowDurationMs: number = 10000, windowOverlap: number = 0.5): Promise<boolean> {
    if (this.isRunning) {
      console.warn('AccelerometerService already running');
      return true;
    }
    
    const available = await this.isAvailable();
    if (!available) {
      console.error('Accelerometer not available or permission denied');
      return false;
    }
    
    this.windowDurationMs = windowDurationMs;
    this.windowOverlap = windowOverlap;
    this.sampleBuffer = [];
    this.lastWindowEndTime = Date.now();
    
    // Set update interval
    Accelerometer.setUpdateInterval(this.config.updateIntervalMs);
    
    // Subscribe to updates
    this.subscription = Accelerometer.addListener(this.handleMeasurement.bind(this));
    this.isRunning = true;
    
    console.log(`AccelerometerService started: ${1000 / this.config.updateIntervalMs} Hz target`);
    return true;
  }
  
  /**
   * Stop capturing
   */
  stop(): void {
    if (this.subscription) {
      this.subscription.remove();
      this.subscription = null;
    }
    this.isRunning = false;
    console.log('AccelerometerService stopped');
  }
  
  /**
   * Handle incoming measurement
   */
  private handleMeasurement(measurement: AccelerometerMeasurement): void {
    const timestamp = Date.now();
    const { x, y, z } = measurement;
    
    const sample: AccelerometerSample = {
      timestamp,
      x,
      y,
      z,
      magnitude: Math.sqrt(x * x + y * y + z * z),
    };
    
    // Notify sample callbacks
    for (const cb of this.sampleCallbacks) {
      cb(sample);
    }
    
    // Add to buffer
    this.sampleBuffer.push(sample);
    
    // Check if window is complete
    this.checkWindowCompletion(timestamp);
  }
  
  /**
   * Check if we have a complete window
   */
  private checkWindowCompletion(currentTime: number): void {
    const windowStepMs = this.windowDurationMs * (1 - this.windowOverlap);
    
    if (currentTime - this.lastWindowEndTime >= windowStepMs) {
      // Extract window
      const windowEndTime = currentTime;
      const windowStartTime = windowEndTime - this.windowDurationMs;
      
      // Filter samples within window
      const windowSamples = this.sampleBuffer.filter(
        s => s.timestamp >= windowStartTime && s.timestamp <= windowEndTime
      );
      
      if (windowSamples.length > 10) { // Minimum samples
        const sampleRate = windowSamples.length / (this.windowDurationMs / 1000);
        
        const window: VibrationWindow = {
          id: `window_${this.windowCounter++}`,
          startTime: windowStartTime,
          endTime: windowEndTime,
          samples: windowSamples,
          sampleRate,
        };
        
        // Notify window callbacks
        for (const cb of this.windowCallbacks) {
          cb(window);
        }
      }
      
      // Update last window time
      this.lastWindowEndTime = windowEndTime;
      
      // Trim old samples (keep overlap)
      const overlapStartTime = currentTime - this.windowDurationMs;
      this.sampleBuffer = this.sampleBuffer.filter(s => s.timestamp >= overlapStartTime);
    }
  }
  
  /**
   * Register callback for each sample
   */
  onSample(callback: AccelerometerCallback): () => void {
    this.sampleCallbacks.push(callback);
    return () => {
      const idx = this.sampleCallbacks.indexOf(callback);
      if (idx >= 0) this.sampleCallbacks.splice(idx, 1);
    };
  }
  
  /**
   * Register callback for each completed window
   */
  onWindow(callback: WindowCallback): () => void {
    this.windowCallbacks.push(callback);
    return () => {
      const idx = this.windowCallbacks.indexOf(callback);
      if (idx >= 0) this.windowCallbacks.splice(idx, 1);
    };
  }
  
  /**
   * Get current buffer
   */
  getBuffer(): AccelerometerSample[] {
    return [...this.sampleBuffer];
  }
  
  /**
   * Check if running
   */
  getIsRunning(): boolean {
    return this.isRunning;
  }
}

export default AccelerometerService;
