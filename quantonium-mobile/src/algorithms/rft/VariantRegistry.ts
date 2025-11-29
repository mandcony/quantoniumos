/**
 * SPDX-License-Identifier: LicenseRef-QuantoniumOS-Claims-NC
 * Copyright (C) 2025 Luis M. Minier / quantoniumos
 * Listed in CLAIMS_PRACTICING_FILES.txt and licensed under LICENSE-CLAIMS-NC.md
 * (research/education only). Commercial rights require a separate patent license.
 */

import { Complex, fromPolar, multiply } from './Complex';
import {
  Matrix,
  matrixFromGenerator,
  qrDecomposition,
  matrixIdentity,
  matrixMultiply,
  matrixConjugateTranspose,
  matrixNorm,
  scaleMatrix,
  addMatrices,
} from './Matrix';
import { CanonicalTrueRFT } from './RFTCore';

const PHI = (1 + Math.sqrt(5)) / 2;
const TWO_PI = 2 * Math.PI;

export interface VariantInfo {
  key: string;
  name: string;
  innovation: string;
  useCase: string;
  generator: (size: number) => Matrix;
}

function orthonormalize(matrix: Matrix): Matrix {
  return qrDecomposition(matrix);
}

function seededRandom(seed: number): () => number {
  let state = seed >>> 0;
  return () => {
    state = (1103515245 * state + 12345) & 0x7fffffff;
    return state / 0x7fffffff;
  };
}

function generateOriginalPhiRFT(size: number): Matrix {
  const invSqrt = 1 / Math.sqrt(size);
  const phiPowers = Array.from({ length: size }, (_, j) => Math.pow(PHI, -j));
  const matrix = matrixFromGenerator(size, (row, col) => {
    const sample = row;
    const phi = phiPowers[col];
    const theta =
      (TWO_PI * phi * sample) / size + (Math.PI * phi * (sample * sample)) / (2 * size);
    return fromPolar(invSqrt, theta);
  });
  return orthonormalize(matrix);
}

function generateHarmonicPhase(size: number, alpha = 0.5): Matrix {
  const invSqrt = 1 / Math.sqrt(size);
  const matrix = matrixFromGenerator(size, (row, col) => {
    const sample = row;
    const phase =
      (TWO_PI * col * sample) / size +
      (alpha * Math.PI * Math.pow(col * sample, 3)) / (size * size);
    return fromPolar(invSqrt, phase);
  });
  return orthonormalize(matrix);
}

function fibonacciSequence(length: number): number[] {
  const fib = [1, 1];
  while (fib.length <= length) {
    fib.push(fib[fib.length - 1] + fib[fib.length - 2]);
  }
  return fib;
}

function generateFibonacciTilt(size: number): Matrix {
  const fib = fibonacciSequence(size + 5);
  const fN = fib[size];
  const invSqrt = 1 / Math.sqrt(size);
  const matrix = matrixFromGenerator(size, (row, col) => {
    const phase = (TWO_PI * fib[col] * row) / fN;
    return fromPolar(invSqrt, phase);
  });
  return orthonormalize(matrix);
}

function generateChaoticMix(size: number, seed = 42): Matrix {
  const rand = seededRandom(seed);
  const matrix = matrixFromGenerator(size, () => {
    const re = rand() * 2 - 1;
    const im = rand() * 2 - 1;
    return { re, im };
  });
  return orthonormalize(matrix);
}

function generateGeometricLattice(size: number): Matrix {
  const invSqrt = 1 / Math.sqrt(size);
  const matrix = matrixFromGenerator(size, (row, col) => {
    const phase =
      (TWO_PI * col * row) / size +
      (TWO_PI * (row * row * col + row * col * col)) / (size * size);
    return fromPolar(invSqrt, phase);
  });
  return orthonormalize(matrix);
}

function generatePhiChaoticHybrid(size: number): Matrix {
  const fibMatrix = generateFibonacciTilt(size);
  const chaosMatrix = generateChaoticMix(size);
  const combined = addMatrices(scaleMatrix(fibMatrix, 1 / Math.sqrt(2)), scaleMatrix(chaosMatrix, 1 / Math.sqrt(2)));
  return orthonormalize(combined);
}

function generateAdaptivePhi(size: number): Matrix {
  return generatePhiChaoticHybrid(size);
}

function dftMatrix(size: number): Matrix {
  const invSqrt = 1 / Math.sqrt(size);
  return matrixFromGenerator(size, (row, col) => {
    const theta = (TWO_PI * row * col) / size;
    return fromPolar(invSqrt, theta);
  });
}

function fftFactorizedBasis(size: number, goldenPhase: Complex[], sigma = 1.25): Matrix {
  const phaseQuadratic = Array.from({ length: size }, (_, k) => {
    const theta = (Math.PI * sigma * k * k) / size;
    return fromPolar(1, theta);
  });
  const fft = dftMatrix(size);
  const scaled = matrixFromGenerator(size, (row, col) => {
    const base = fft.data[row * size + col];
    const withPhase = multiply(goldenPhase[row], base);
    return multiply(phaseQuadratic[row], withPhase);
  });
  return orthonormalize(scaled);
}

function generateLogPeriodicPhiRFT(size: number, beta = 0.83, sigma = 1.25): Matrix {
  const goldenPhase = Array.from({ length: size }, (_, k) => {
    const logk = Math.log1p(k) / Math.log1p(size);
    const theta = TWO_PI * beta * logk;
    return fromPolar(1, theta);
  });
  return fftFactorizedBasis(size, goldenPhase, sigma);
}

function generateConvexMixedPhiRFT(
  size: number,
  beta = 0.83,
  sigma = 1.25,
  mix = 0.5,
): Matrix {
  const mixValue = Math.min(Math.max(mix, 0), 1);
  const goldenPhase = Array.from({ length: size }, (_, k) => {
    const frac = (k / PHI) % 1;
    const logk = Math.log1p(k) / Math.log1p(size);
    const theta = (1 - mixValue) * (TWO_PI * beta * frac) + mixValue * (TWO_PI * beta * logk);
    return fromPolar(1, theta);
  });
  return fftFactorizedBasis(size, goldenPhase, sigma);
}

function generateExactGoldenRatioUnitary(size: number): Matrix {
  const rft = new CanonicalTrueRFT(size);
  return rft.getRFTMatrix();
}

export const VARIANTS: VariantInfo[] = [
  {
    key: 'original',
    name: 'Original Φ-RFT',
    innovation: 'Golden-resonant phase',
    useCase: 'Quantum simulation',
    generator: generateOriginalPhiRFT,
  },
  {
    key: 'harmonic_phase',
    name: 'Harmonic-Phase',
    innovation: 'Cubic time-base',
    useCase: 'Nonlinear filtering',
    generator: generateHarmonicPhase,
  },
  {
    key: 'fibonacci_tilt',
    name: 'Fibonacci Tilt',
    innovation: 'Integer lattice alignment',
    useCase: 'Lattice structures (experimental)',
    generator: generateFibonacciTilt,
  },
  {
    key: 'chaotic_mix',
    name: 'Chaotic Mix',
    innovation: 'Haar-like randomness',
    useCase: 'Mixing/diffusion (experimental)',
    generator: generateChaoticMix,
  },
  {
    key: 'geometric_lattice',
    name: 'Geometric Lattice',
    innovation: 'Phase-engineered lattice',
    useCase: 'Analog / optical computing',
    generator: generateGeometricLattice,
  },
  {
    key: 'phi_chaotic_hybrid',
    name: 'Φ-Chaotic Hybrid',
    innovation: 'Structure + disorder',
    useCase: 'Resilient codecs',
    generator: generatePhiChaoticHybrid,
  },
  {
    key: 'adaptive_phi',
    name: 'Adaptive Φ',
    innovation: 'Meta selection',
    useCase: 'Universal compression',
    generator: generateAdaptivePhi,
  },
  {
    key: 'log_periodic',
    name: 'Log-Periodic Φ-RFT',
    innovation: 'Log-frequency phase warp',
    useCase: 'Symbol compression',
    generator: generateLogPeriodicPhiRFT,
  },
  {
    key: 'convex_mix',
    name: 'Convex Mixed Φ-RFT',
    innovation: 'Hybrid log/standard phase',
    useCase: 'Adaptive textures',
    generator: generateConvexMixedPhiRFT,
  },
  {
    key: 'golden_ratio_exact',
    name: 'Exact Golden Ratio Kernel',
    innovation: 'Full resonance lattice',
    useCase: 'Theorem validation',
    generator: generateExactGoldenRatioUnitary,
  },
];

export function validateVariantMatrix(matrix: Matrix): number {
  const identity = matrixIdentity(matrix.rows);
  const product = matrixMultiply(matrixConjugateTranspose(matrix), matrix);
  return matrixNorm(product, identity);
}
