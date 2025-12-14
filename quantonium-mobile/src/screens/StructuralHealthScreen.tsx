/**
 * Structural Health Monitor Screen
 * RFT-based vibration monitoring with auto-calibration
 */

import React, { useState, useEffect, useRef, useCallback } from 'react';
import {
  View,
  Text,
  StyleSheet,
  ScrollView,
  Switch,
  TouchableOpacity,
  Dimensions,
  Alert,
  Share,
} from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import { StatusBar } from 'expo-status-bar';
import { Accelerometer } from 'expo-sensors';

import {
  SHMEngine,
  SHMResult,
  SHMDomain,
} from '../algorithms/shm/SHMEngine';

const { width: SCREEN_WIDTH } = Dimensions.get('window');
const CHART_WIDTH = SCREEN_WIDTH - 80;
const CHART_HEIGHT = 140;
const MAX_DATA_POINTS = 50;
const SAMPLE_RATE = 50; // Hz
const CALIBRATION_SAMPLES = 100; // 2 seconds at 50Hz

const DOMAINS: { key: SHMDomain; emoji: string; label: string; desc: string }[] = [
  { key: 'seismic', emoji: '🌍', label: 'Seismic', desc: 'Earthquake & ground motion' },
  { key: 'bridge', emoji: '🌉', label: 'Bridge', desc: 'Deck vibration & modal tracking' },
  { key: 'building', emoji: '🏢', label: 'Building', desc: 'Sway & occupant comfort' },
  { key: 'construction', emoji: '🏗️', label: 'Construction', desc: 'PPV compliance (DIN 4150)' },
  { key: 'industrial', emoji: '⚙️', label: 'Industrial', desc: 'Machine health (ISO 10816)' },
];

/**
 * Calibration data - orientation independent
 * We store the gravity vector direction, not just offsets
 */
interface CalibrationData {
  // Mean gravity vector (direction device is oriented)
  gravityX: number;
  gravityY: number;
  gravityZ: number;
  gravityMagnitude: number;
  
  // Sensor bias (drift from true values)
  biasX: number;
  biasY: number;
  biasZ: number;
  
  // Noise characteristics
  noiseFloorRMS: number;    // RMS of vibration during calibration
  noiseFloorPeak: number;   // Peak noise during calibration
  
  // Quality metrics
  sampleCount: number;
  variance: number;         // Variance during calibration (stability indicator)
  isStable: boolean;        // Was device stable during calibration?
  
  timestamp: number;
}

export default function StructuralHealthScreen() {
  // Engine
  const engineRef = useRef<SHMEngine | null>(null);
  
  // State
  const [selectedDomain, setSelectedDomain] = useState<SHMDomain>('building');
  const [isCalibrating, setIsCalibrating] = useState(false);
  const [isCalibrated, setIsCalibrated] = useState(false);
  const [isMonitoring, setIsMonitoring] = useState(false);
  const [calibrationProgress, setCalibrationProgress] = useState(0);
  
  // Calibration data
  const calibrationRef = useRef<CalibrationData | null>(null);
  const calibrationSamplesRef = useRef<{ x: number; y: number; z: number }[]>([]);
  
  // Measurement data
  const [currentResult, setCurrentResult] = useState<SHMResult | null>(null);
  const [vibrationHistory, setVibrationHistory] = useState<number[]>([]);
  const [alerts, setAlerts] = useState<string[]>([]);
  const [status, setStatus] = useState<'idle' | 'calibrating' | 'stable' | 'warning' | 'critical'>('idle');
  
  // Sample buffer for analysis
  const sampleBufferRef = useRef<{ timestamp: number; x: number; y: number; z: number; magnitude: number }[]>([]);
  
  // Subscription
  const subscriptionRef = useRef<ReturnType<typeof Accelerometer.addListener> | null>(null);
  
  // Initialize engine
  useEffect(() => {
    engineRef.current = new SHMEngine(selectedDomain, SAMPLE_RATE);
  }, []);
  
  // Handle domain change
  const handleDomainChange = useCallback((domain: SHMDomain) => {
    if (isMonitoring || isCalibrating) return;
    
    setSelectedDomain(domain);
    if (engineRef.current) {
      engineRef.current.setDomain(domain);
    }
    // Keep calibration valid - it's device-specific, not domain-specific
  }, [isMonitoring, isCalibrating]);
  
  // Complete calibration - compute gravity vector, bias, and noise floor
  const completeCalibration = useCallback(() => {
    if (subscriptionRef.current) {
      subscriptionRef.current.remove();
      subscriptionRef.current = null;
    }
    
    const samples = calibrationSamplesRef.current;
    if (samples.length < 50) {
      setIsCalibrating(false);
      setStatus('idle');
      Alert.alert('Calibration Failed', 'Not enough samples collected. Try again.');
      return;
    }
    
    const n = samples.length;
    
    // ========================================
    // Step 1: Compute mean (gravity vector)
    // ========================================
    const sumX = samples.reduce((a, s) => a + s.x, 0);
    const sumY = samples.reduce((a, s) => a + s.y, 0);
    const sumZ = samples.reduce((a, s) => a + s.z, 0);
    
    const meanX = sumX / n;
    const meanY = sumY / n;
    const meanZ = sumZ / n;
    
    // Gravity magnitude (should be ~1.0g if sensor is accurate)
    const gravityMagnitude = Math.sqrt(meanX ** 2 + meanY ** 2 + meanZ ** 2);
    
    // Normalize to get gravity direction (unit vector)
    const gravityX = meanX / gravityMagnitude;
    const gravityY = meanY / gravityMagnitude;
    const gravityZ = meanZ / gravityMagnitude;
    
    // ========================================
    // Step 2: Compute sensor bias
    // ========================================
    // Bias is how far the measured gravity is from expected 1.0g
    // If gravityMagnitude != 1.0, there's sensor scale error
    const biasX = meanX - gravityX;  // Component beyond unit gravity
    const biasY = meanY - gravityY;
    const biasZ = meanZ - gravityZ;
    
    // ========================================
    // Step 3: Compute vibration during calibration
    // ========================================
    // Project each sample onto gravity direction and subtract
    // What remains is the vibration component
    const vibrations: number[] = [];
    
    for (const s of samples) {
      // Project sample onto gravity vector (dot product)
      const gravityComponent = s.x * gravityX + s.y * gravityY + s.z * gravityZ;
      
      // Subtract gravity to get vibration-only components
      const vibX = s.x - gravityComponent * gravityX;
      const vibY = s.y - gravityComponent * gravityY;
      const vibZ = s.z - gravityComponent * gravityZ;
      
      // Also include any variation in gravity direction (for stability check)
      const totalVibration = Math.sqrt(vibX ** 2 + vibY ** 2 + vibZ ** 2);
      vibrations.push(totalVibration);
    }
    
    // RMS noise floor
    const sumSqVib = vibrations.reduce((a, v) => a + v ** 2, 0);
    const noiseFloorRMS = Math.sqrt(sumSqVib / n);
    
    // Peak noise
    const noiseFloorPeak = Math.max(...vibrations);
    
    // ========================================
    // Step 4: Stability check
    // ========================================
    // Compute variance to detect if device was moving during calibration
    let sumSqDeviation = 0;
    for (const s of samples) {
      sumSqDeviation += (s.x - meanX) ** 2 + (s.y - meanY) ** 2 + (s.z - meanZ) ** 2;
    }
    const variance = sumSqDeviation / n;
    
    // Device is stable if variance is low (< 0.001 g²)
    const isStable = variance < 0.001;
    
    // ========================================
    // Step 5: Store calibration
    // ========================================
    calibrationRef.current = {
      gravityX,
      gravityY,
      gravityZ,
      gravityMagnitude,
      biasX,
      biasY,
      biasZ,
      noiseFloorRMS,
      noiseFloorPeak,
      sampleCount: n,
      variance,
      isStable,
      timestamp: Date.now(),
    };
    
    setIsCalibrating(false);
    setIsCalibrated(true);
    setStatus('stable');
    
    if (!isStable) {
      Alert.alert(
        '⚠️ Calibration Complete (Unstable)',
        `Device moved during calibration.\nNoise: ${(noiseFloorRMS * 1000).toFixed(1)} mg\nFor best results, keep device still and recalibrate.`
      );
    } else {
      Alert.alert(
        '✅ Calibration Complete',
        `Sensor calibrated.\n` +
        `Noise floor: ${(noiseFloorRMS * 1000).toFixed(2)} mg\n` +
        `Gravity: ${gravityMagnitude.toFixed(3)}g\n` +
        `Samples: ${n}\n` +
        `Ready to monitor.`
      );
    }
  }, []);
  
  // Start calibration
  const startCalibration = useCallback(async () => {
    try {
      const { status } = await Accelerometer.requestPermissionsAsync();
      if (status !== 'granted') {
        Alert.alert('Permission Denied', 'Accelerometer access is required.');
        return;
      }
      
      setIsCalibrating(true);
      setStatus('calibrating');
      setCalibrationProgress(0);
      calibrationSamplesRef.current = [];
      
      Accelerometer.setUpdateInterval(1000 / SAMPLE_RATE);
      
      subscriptionRef.current = Accelerometer.addListener(({ x, y, z }) => {
        calibrationSamplesRef.current.push({ x, y, z });
        setCalibrationProgress(Math.min(100, (calibrationSamplesRef.current.length / CALIBRATION_SAMPLES) * 100));
        
        if (calibrationSamplesRef.current.length >= CALIBRATION_SAMPLES) {
          completeCalibration();
        }
      });
    } catch (e) {
      console.error('Calibration failed:', e);
      setIsCalibrating(false);
      setStatus('idle');
    }
  }, [completeCalibration]);
  
  // Process a window of samples
  const processWindow = useCallback(() => {
    if (!engineRef.current || sampleBufferRef.current.length < 32) return;
    
    const samples = [...sampleBufferRef.current];
    sampleBufferRef.current = [];
    
    try {
      const result = engineRef.current.analyze(samples);
      setCurrentResult(result);
      
      // Update history
      const rms = Math.sqrt(samples.reduce((s, x) => s + x.magnitude ** 2, 0) / samples.length);
      setVibrationHistory(prev => [...prev, rms].slice(-MAX_DATA_POINTS));
      
      // Update alerts
      if (result.alerts.length > 0) {
        setAlerts(prev => [...result.alerts, ...prev].slice(0, 10));
        const hasCritical = result.alerts.some(a => a.includes('🚨') || a.includes('CRITICAL'));
        setStatus(hasCritical ? 'critical' : 'warning');
      } else {
        setStatus('stable');
      }
    } catch (e) {
      console.error('Analysis error:', e);
    }
  }, []);
  
  // Start monitoring
  const startMonitoring = useCallback(async () => {
    if (!isCalibrated || !calibrationRef.current) {
      Alert.alert('Calibrate First', 'Please calibrate the sensor before monitoring.');
      return;
    }
    
    try {
      Accelerometer.setUpdateInterval(1000 / SAMPLE_RATE);
      sampleBufferRef.current = [];
      
      const cal = calibrationRef.current;
      
      subscriptionRef.current = Accelerometer.addListener(({ x, y, z }) => {
        // ========================================
        // Orientation-independent gravity removal
        // ========================================
        
        // Raw accelerometer reading
        const rawX = x;
        const rawY = y;
        const rawZ = z;
        
        // Project onto gravity vector (dot product with unit gravity)
        const gravityComponent = rawX * cal.gravityX + rawY * cal.gravityY + rawZ * cal.gravityZ;
        
        // Remove gravity component from each axis
        // This leaves only the vibration/dynamic acceleration
        const vibX = rawX - gravityComponent * cal.gravityX;
        const vibY = rawY - gravityComponent * cal.gravityY;
        const vibZ = rawZ - gravityComponent * cal.gravityZ;
        
        // Compute vibration magnitude (gravity already removed)
        const magnitude = Math.sqrt(vibX ** 2 + vibY ** 2 + vibZ ** 2);
        
        // Only record if above noise floor (prevents recording sensor noise)
        // But always record for time continuity in analysis
        const timestamp = Date.now();
        
        sampleBufferRef.current.push({
          timestamp,
          x: vibX,
          y: vibY,
          z: vibZ,
          magnitude,
        });
        
        // Process every 1 second (SAMPLE_RATE samples)
        if (sampleBufferRef.current.length >= SAMPLE_RATE) {
          processWindow();
        }
      });
      
      setIsMonitoring(true);
      setStatus('stable');
    } catch (e) {
      console.error('Failed to start monitoring:', e);
    }
  }, [isCalibrated, processWindow]);
  
  // Stop monitoring
  const stopMonitoring = useCallback(() => {
    if (subscriptionRef.current) {
      subscriptionRef.current.remove();
      subscriptionRef.current = null;
    }
    setIsMonitoring(false);
  }, []);
  
  // Toggle monitoring
  const toggleMonitoring = useCallback((value: boolean) => {
    if (value) {
      startMonitoring();
    } else {
      stopMonitoring();
    }
  }, [startMonitoring, stopMonitoring]);
  
  // Export data
  const exportData = useCallback(async () => {
    if (!currentResult) {
      Alert.alert('No Data', 'Start monitoring to collect data first.');
      return;
    }
    
    const exportObj = {
      domain: selectedDomain,
      calibration: calibrationRef.current,
      lastResult: currentResult,
      timestamp: new Date().toISOString(),
    };
    
    try {
      await Share.share({
        message: JSON.stringify(exportObj, null, 2),
        title: `SHM Export - ${selectedDomain}`,
      });
    } catch (e) {
      Alert.alert('Export Failed', String(e));
    }
  }, [selectedDomain, currentResult]);
  
  // Cleanup
  useEffect(() => {
    return () => {
      if (subscriptionRef.current) {
        subscriptionRef.current.remove();
      }
    };
  }, []);
  
  // Get domain-specific display
  const getDomainMetrics = () => {
    if (!currentResult) return null;
    
    switch (selectedDomain) {
      case 'seismic':
        return currentResult.seismic ? (
          <>
            <MetricItem label="PGA" value={`${(currentResult.seismic.pga * 1000).toFixed(1)} mg`} />
            <MetricItem label="PGV" value={`${(currentResult.seismic.pgv * 1000).toFixed(2)} mm/s`} />
            <MetricItem label="Intensity" value={currentResult.seismic.mercalli} highlight />
            <MetricItem label="CAV" value={`${currentResult.seismic.cav.toFixed(3)} m/s`} />
            <MetricItem label="Arias" value={`${currentResult.seismic.arias.toFixed(4)} m/s`} />
            <MetricItem label="Freq" value={`${currentResult.seismic.dominantFrequency.toFixed(1)} Hz`} />
          </>
        ) : null;
        
      case 'bridge':
        return currentResult.bridge ? (
          <>
            <MetricItem label="Deck Freq" value={`${currentResult.bridge.deckFrequency.toFixed(2)} Hz`} />
            <MetricItem label="Damping" value={`${currentResult.bridge.dampingRatio.toFixed(1)}%`} />
            <MetricItem label="Health" value={`${currentResult.bridge.healthIndex.toFixed(0)}/100`} highlight />
            <MetricItem label="Freq Shift" value={`${currentResult.bridge.frequencyShift.toFixed(1)}%`} />
            <MetricItem label="Traffic" value={`${(currentResult.bridge.trafficIndex * 100).toFixed(0)}%`} />
            <MetricItem label="Accel" value={`${currentResult.bridge.deckAcceleration.toFixed(3)} m/s²`} />
          </>
        ) : null;
        
      case 'building':
        return currentResult.building ? (
          <>
            <MetricItem label="Sway Freq" value={`${currentResult.building.swayFrequency.toFixed(2)} Hz`} />
            <MetricItem label="Sway" value={`${currentResult.building.swayAmplitude.toFixed(1)} mm`} />
            <MetricItem label="Comfort" value={currentResult.building.comfortLevel} highlight />
            <MetricItem label="Damping" value={`${currentResult.building.dampingRatio.toFixed(1)}%`} />
            <MetricItem label="RMS" value={`${(currentResult.building.accelerationRMS * 1000).toFixed(1)} mg`} />
            <MetricItem label="Perception" value={currentResult.building.humanPerception} />
          </>
        ) : null;
        
      case 'construction':
        return currentResult.construction ? (
          <>
            <MetricItem label="PPV" value={`${currentResult.construction.ppv.toFixed(1)} mm/s`} highlight />
            <MetricItem label="Status" value={currentResult.construction.complianceStatus} />
            <MetricItem label="Margin" value={`${currentResult.construction.margin.toFixed(0)}%`} />
            <MetricItem label="VDV" value={`${currentResult.construction.vibrationDose.toFixed(3)}`} />
            <MetricItem label="Freq" value={`${currentResult.construction.frequency.toFixed(1)} Hz`} />
            <MetricItem label="Source" value={currentResult.construction.sourceType} />
          </>
        ) : null;
        
      case 'industrial':
        return currentResult.machine ? (
          <>
            <MetricItem label="RPM" value={`${currentResult.machine.rpm.toFixed(0)}`} />
            <MetricItem label="Vibration" value={`${currentResult.machine.overallVibration.toFixed(2)} mm/s`} />
            <MetricItem label="Bearing" value={currentResult.machine.bearingCondition} highlight />
            <MetricItem label="Health" value={`${currentResult.machine.healthScore.toFixed(0)}/100`} />
            <MetricItem label="Life" value={currentResult.machine.remainingLife} />
            <MetricItem label="1X Freq" value={`${currentResult.machine.fundamentalFrequency.toFixed(1)} Hz`} />
          </>
        ) : null;
        
      default:
        return null;
    }
  };
  
  // Chart calculations
  const maxVibration = Math.max(0.01, ...vibrationHistory);
  const currentDomain = DOMAINS.find(d => d.key === selectedDomain)!;
  
  return (
    <SafeAreaView style={styles.container}>
      <StatusBar style="dark" />
      <ScrollView style={styles.scrollView} showsVerticalScrollIndicator={false}>
        {/* Header */}
        <View style={styles.header}>
          <Text style={styles.title}>🛠️ Structural Health</Text>
          <Text style={styles.subtitle}>Φ-RFT Vibration Analysis</Text>
        </View>
        
        {/* Domain Selector */}
        <View style={styles.card}>
          <Text style={styles.cardTitle}>Domain</Text>
          <ScrollView horizontal showsHorizontalScrollIndicator={false}>
            <View style={styles.domainSelector}>
              {DOMAINS.map(d => (
                <TouchableOpacity
                  key={d.key}
                  style={[
                    styles.domainButton,
                    selectedDomain === d.key && styles.domainButtonActive,
                  ]}
                  onPress={() => handleDomainChange(d.key)}
                  disabled={isMonitoring || isCalibrating}
                >
                  <Text style={styles.domainEmoji}>{d.emoji}</Text>
                  <Text style={[
                    styles.domainLabel,
                    selectedDomain === d.key && styles.domainLabelActive,
                  ]}>{d.label}</Text>
                </TouchableOpacity>
              ))}
            </View>
          </ScrollView>
          <Text style={styles.domainDescription}>{currentDomain.desc}</Text>
        </View>
        
        {/* Status & Calibration */}
        <View style={[
          styles.card,
          status === 'warning' && styles.cardWarning,
          status === 'critical' && styles.cardCritical,
        ]}>
          <View style={styles.statusRow}>
            <View>
              <Text style={styles.cardTitle}>Status</Text>
              <Text style={[
                styles.statusText,
                status === 'calibrating' && styles.statusCalibrating,
                status === 'warning' && styles.statusWarning,
                status === 'critical' && styles.statusCritical,
              ]}>
                {status === 'idle' && '⏸️ Idle'}
                {status === 'calibrating' && '🔄 Calibrating...'}
                {status === 'stable' && '✅ Stable'}
                {status === 'warning' && '⚠️ Warning'}
                {status === 'critical' && '🚨 Critical'}
              </Text>
            </View>
            
            {!isCalibrated && !isCalibrating && (
              <TouchableOpacity style={styles.calibrateButton} onPress={startCalibration}>
                <Text style={styles.calibrateButtonText}>🎯 Calibrate</Text>
              </TouchableOpacity>
            )}
            
            {isCalibrated && !isMonitoring && (
              <TouchableOpacity 
                style={[styles.calibrateButton, styles.recalibrateButton]} 
                onPress={startCalibration}
              >
                <Text style={styles.recalibrateButtonText}>↻ Recalibrate</Text>
              </TouchableOpacity>
            )}
          </View>
          
          {isCalibrating && (
            <View style={styles.calibrationProgress}>
              <View style={[styles.progressBar, { width: `${calibrationProgress}%` }]} />
              <Text style={styles.progressText}>Keep device still... {calibrationProgress.toFixed(0)}%</Text>
            </View>
          )}
          
          {isCalibrated && calibrationRef.current && (
            <View style={styles.calibrationInfo}>
              <Text style={styles.calibrationText}>
                ✓ Noise: {(calibrationRef.current.noiseFloorRMS * 1000).toFixed(2)} mg RMS
              </Text>
              <Text style={styles.calibrationText}>
                ✓ Gravity: {calibrationRef.current.gravityMagnitude.toFixed(3)}g
              </Text>
              <Text style={[
                styles.calibrationText,
                !calibrationRef.current.isStable && styles.calibrationWarning
              ]}>
                {calibrationRef.current.isStable ? '✓ Stable' : '⚠ Unstable - recalibrate'}
              </Text>
            </View>
          )}
        </View>
        
        {/* Monitoring Toggle */}
        {isCalibrated && (
          <View style={styles.card}>
            <View style={styles.toggleRow}>
              <Text style={styles.toggleLabel}>
                {isMonitoring ? '📡 Monitoring Active' : '📡 Start Monitoring'}
              </Text>
              <Switch
                value={isMonitoring}
                onValueChange={toggleMonitoring}
                trackColor={{ false: '#ddd', true: '#4CAF50' }}
                disabled={isCalibrating}
              />
            </View>
          </View>
        )}
        
        {/* Domain Metrics */}
        {isMonitoring && currentResult && (
          <View style={styles.card}>
            <Text style={styles.cardTitle}>{currentDomain.emoji} {currentDomain.label} Metrics</Text>
            <View style={styles.metricsGrid}>
              {getDomainMetrics()}
            </View>
          </View>
        )}
        
        {/* Vibration Chart */}
        {vibrationHistory.length > 0 && (
          <View style={styles.chartCard}>
            <Text style={styles.chartTitle}>Vibration (RMS)</Text>
            <View style={styles.chartArea}>
              {/* Grid lines */}
              {[0.25, 0.5, 0.75].map((ratio, i) => (
                <View key={i} style={[styles.gridLine, { bottom: ratio * CHART_HEIGHT }]} />
              ))}
              
              {/* Data points */}
              {vibrationHistory.map((val, i) => {
                const x = (i / MAX_DATA_POINTS) * CHART_WIDTH;
                const y = (val / maxVibration) * CHART_HEIGHT;
                
                return (
                  <View
                    key={i}
                    style={[
                      styles.dataPoint,
                      {
                        left: x,
                        bottom: Math.min(y, CHART_HEIGHT - 3),
                        backgroundColor: val > maxVibration * 0.8 ? '#FF6B6B' : '#4CAF50',
                      },
                    ]}
                  />
                );
              })}
            </View>
            <Text style={styles.chartLabel}>
              Current: {(vibrationHistory[vibrationHistory.length - 1] * 1000).toFixed(1)} mg | Max: {(maxVibration * 1000).toFixed(1)} mg
            </Text>
          </View>
        )}
        
        {/* Spectral Info */}
        {currentResult && (
          <View style={styles.card}>
            <Text style={styles.cardTitle}>📊 Spectral Analysis</Text>
            <View style={styles.metricsGrid}>
              <MetricItem label="Centroid" value={`${currentResult.spectral.spectralCentroid.toFixed(1)} Hz`} />
              <MetricItem label="Rolloff" value={`${currentResult.spectral.spectralRolloff.toFixed(1)} Hz`} />
              <MetricItem label="Entropy" value={`${(currentResult.spectral.spectralEntropy * 100).toFixed(0)}%`} />
              <MetricItem label="RFT Sparsity" value={`${(currentResult.spectral.rftSparsity * 100).toFixed(0)}%`} />
              <MetricItem label="Peak Freq" value={`${(currentResult.spectral.peakFrequencies[0] ?? 0).toFixed(1)} Hz`} />
              <MetricItem label="Confidence" value={`${(currentResult.confidence * 100).toFixed(0)}%`} />
            </View>
          </View>
        )}
        
        {/* Alerts */}
        {alerts.length > 0 && (
          <View style={styles.card}>
            <Text style={styles.cardTitle}>🚨 Alerts</Text>
            {alerts.slice(0, 5).map((alert, i) => (
              <View key={i} style={styles.alertRow}>
                <Text style={styles.alertText}>{alert}</Text>
              </View>
            ))}
            {alerts.length > 5 && (
              <Text style={styles.moreAlerts}>+{alerts.length - 5} more</Text>
            )}
          </View>
        )}
        
        {/* Export */}
        {isMonitoring && (
          <TouchableOpacity style={styles.exportButton} onPress={exportData}>
            <Text style={styles.exportButtonText}>📤 Export Data</Text>
          </TouchableOpacity>
        )}
        
        {/* Info */}
        <View style={styles.infoCard}>
          <Text style={styles.infoTitle}>ℹ️ How to Use</Text>
          <Text style={styles.infoText}>
            1. Select your monitoring domain{'\n'}
            2. Place device on stable surface{'\n'}
            3. Tap Calibrate (keep still for 2s){'\n'}
            4. Enable monitoring{'\n'}
            5. View real-time analysis
          </Text>
        </View>
        
        <View style={{ height: 40 }} />
      </ScrollView>
    </SafeAreaView>
  );
}

function MetricItem({ label, value, highlight }: { label: string; value: string; highlight?: boolean }) {
  return (
    <View style={styles.metricItem}>
      <Text style={styles.metricLabel}>{label}</Text>
      <Text style={[styles.metricValue, highlight && styles.metricHighlight]}>{value}</Text>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#f5f5f5',
  },
  scrollView: {
    flex: 1,
    paddingHorizontal: 16,
  },
  header: {
    paddingVertical: 12,
    alignItems: 'center',
  },
  title: {
    fontSize: 24,
    fontWeight: 'bold',
    color: '#333',
  },
  subtitle: {
    fontSize: 12,
    color: '#666',
    marginTop: 2,
  },
  card: {
    backgroundColor: '#fff',
    borderRadius: 12,
    padding: 16,
    marginBottom: 12,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 1 },
    shadowOpacity: 0.1,
    shadowRadius: 2,
    elevation: 2,
  },
  cardWarning: {
    borderLeftWidth: 4,
    borderLeftColor: '#FF9800',
  },
  cardCritical: {
    borderLeftWidth: 4,
    borderLeftColor: '#F44336',
  },
  cardTitle: {
    fontSize: 14,
    fontWeight: '600',
    color: '#333',
    marginBottom: 8,
  },
  domainSelector: {
    flexDirection: 'row',
    paddingVertical: 4,
  },
  domainButton: {
    alignItems: 'center',
    paddingVertical: 8,
    paddingHorizontal: 12,
    borderRadius: 8,
    backgroundColor: '#f0f0f0',
    marginRight: 8,
    minWidth: 70,
  },
  domainButtonActive: {
    backgroundColor: '#E3F2FD',
    borderWidth: 1,
    borderColor: '#2196F3',
  },
  domainEmoji: {
    fontSize: 24,
  },
  domainLabel: {
    fontSize: 11,
    color: '#666',
    marginTop: 4,
  },
  domainLabelActive: {
    color: '#2196F3',
    fontWeight: '600',
  },
  domainDescription: {
    fontSize: 12,
    color: '#888',
    textAlign: 'center',
    marginTop: 8,
  },
  statusRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
  },
  statusText: {
    fontSize: 18,
    fontWeight: 'bold',
    color: '#4CAF50',
  },
  statusCalibrating: {
    color: '#2196F3',
  },
  statusWarning: {
    color: '#FF9800',
  },
  statusCritical: {
    color: '#F44336',
  },
  calibrateButton: {
    backgroundColor: '#2196F3',
    paddingHorizontal: 16,
    paddingVertical: 10,
    borderRadius: 8,
  },
  calibrateButtonText: {
    color: '#fff',
    fontWeight: '600',
    fontSize: 14,
  },
  recalibrateButton: {
    backgroundColor: '#fff',
    borderWidth: 1,
    borderColor: '#2196F3',
  },
  recalibrateButtonText: {
    color: '#2196F3',
    fontWeight: '600',
    fontSize: 14,
  },
  calibrationProgress: {
    marginTop: 12,
    height: 24,
    backgroundColor: '#E3F2FD',
    borderRadius: 12,
    overflow: 'hidden',
    justifyContent: 'center',
  },
  progressBar: {
    position: 'absolute',
    left: 0,
    top: 0,
    bottom: 0,
    backgroundColor: '#2196F3',
    borderRadius: 12,
  },
  progressText: {
    textAlign: 'center',
    fontSize: 12,
    color: '#333',
    fontWeight: '500',
  },
  calibrationInfo: {
    marginTop: 10,
    padding: 8,
    backgroundColor: '#f8f8f8',
    borderRadius: 6,
  },
  calibrationText: {
    fontSize: 11,
    color: '#666',
    marginBottom: 2,
  },
  calibrationWarning: {
    color: '#FF9800',
    fontWeight: '600',
  },
  toggleRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
  },
  toggleLabel: {
    fontSize: 16,
    color: '#333',
    fontWeight: '500',
  },
  metricsGrid: {
    flexDirection: 'row',
    flexWrap: 'wrap',
  },
  metricItem: {
    width: '50%',
    paddingVertical: 6,
  },
  metricLabel: {
    fontSize: 11,
    color: '#888',
  },
  metricValue: {
    fontSize: 16,
    color: '#333',
    fontWeight: '500',
  },
  metricHighlight: {
    color: '#2196F3',
    fontWeight: '700',
  },
  chartCard: {
    backgroundColor: '#fff',
    borderRadius: 12,
    padding: 16,
    marginBottom: 12,
  },
  chartTitle: {
    fontSize: 14,
    fontWeight: '600',
    color: '#333',
    marginBottom: 8,
  },
  chartArea: {
    height: CHART_HEIGHT,
    backgroundColor: '#fafafa',
    borderRadius: 8,
    position: 'relative',
    overflow: 'hidden',
  },
  gridLine: {
    position: 'absolute',
    left: 0,
    right: 0,
    height: 1,
    backgroundColor: '#eee',
  },
  dataPoint: {
    position: 'absolute',
    width: 6,
    height: 6,
    borderRadius: 3,
    marginLeft: -3,
    marginBottom: -3,
  },
  chartLabel: {
    fontSize: 11,
    color: '#666',
    textAlign: 'center',
    marginTop: 8,
  },
  alertRow: {
    padding: 8,
    backgroundColor: '#FFF3E0',
    borderRadius: 6,
    marginBottom: 4,
  },
  alertText: {
    fontSize: 12,
    color: '#E65100',
  },
  moreAlerts: {
    fontSize: 11,
    color: '#888',
    textAlign: 'center',
    marginTop: 4,
  },
  exportButton: {
    backgroundColor: '#fff',
    borderWidth: 1,
    borderColor: '#2196F3',
    borderRadius: 8,
    padding: 14,
    alignItems: 'center',
    marginBottom: 12,
  },
  exportButtonText: {
    color: '#2196F3',
    fontWeight: '600',
    fontSize: 14,
  },
  infoCard: {
    backgroundColor: '#E8F5E9',
    borderRadius: 12,
    padding: 16,
    marginBottom: 12,
  },
  infoTitle: {
    fontSize: 14,
    fontWeight: '600',
    color: '#2E7D32',
    marginBottom: 8,
  },
  infoText: {
    fontSize: 12,
    color: '#2E7D32',
    lineHeight: 20,
  },
});
