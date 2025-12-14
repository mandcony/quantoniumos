/**
 * Structural Health Monitor - Algorithm exports
 * 
 * Three implementations available:
 * 1. SHMEngine - Production-grade multi-domain engine (recommended)
 * 2. SHMAnalyzer - RFT-based analyzer with baseline comparison
 * 3. VibrationAnalyzer - Original implementation
 */

// Production engine (recommended)
export {
  SHMEngine,
  DIN_4150_LIMITS,
  HUMAN_PERCEPTION,
  TYPICAL_FREQUENCIES,
  MERCALLI_INTENSITY,
  GRAVITY,
  type SHMDomain,
  type AccelerometerSample,
  type ModalParameters,
  type SeismicEvent,
  type BridgeHealth,
  type BuildingResponse,
  type ConstructionCompliance,
  type MachineHealth,
  type SpectralAnalysis,
  type SHMResult,
} from './SHMEngine';

// RFT-based analyzer
export {
  SHMAnalyzer,
  DOMAIN_CONFIGS,
  buildRFTKernel,
  type SHMDomain as SHMAnalyzerDomain,
  type DomainConfig,
  type SHMFeatures,
  type SHMAlert,
} from './SHMAnalyzer';

// Original implementation
export { 
  VibrationAnalyzer,
  DEFAULT_SHM_CONFIG,
  type SHMConfig,
  type AccelerometerSample as VibrationSample,
  type VibrationWindow,
  type SpectralFeatures,
  type BaselineModel,
  type AnomalyScore,
  type SHMEvent,
} from './VibrationAnalyzer';

// Accelerometer service
export {
  AccelerometerService,
  DEFAULT_ACCELEROMETER_CONFIG,
  type AccelerometerConfig,
  type AccelerometerCallback,
  type WindowCallback,
} from './AccelerometerService';
