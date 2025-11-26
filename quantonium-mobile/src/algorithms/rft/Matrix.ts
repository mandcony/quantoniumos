/**
 * SPDX-License-Identifier: LicenseRef-QuantoniumOS-Claims-NC
 * Copyright (C) 2025 Luis M. Minier / quantoniumos
 * Listed in CLAIMS_PRACTICING_FILES.txt and licensed under LICENSE-CLAIMS-NC.md
 * (research/education only). Commercial rights require a separate patent license.
 */

/**
 * Matrix Operations for Complex Matrices
 * Essential linear algebra operations for RFT and quantum algorithms
 */

import { Complex, add, subtract, multiply, conj, abs, divide, scale } from './Complex';

export interface Matrix {
  rows: number;
  cols: number;
  data: Complex[]; // Row-major order
}

/**
 * Create identity matrix
 */
export function matrixIdentity(size: number): Matrix {
  const data: Complex[] = [];
  for (let i = 0; i < size; i++) {
    for (let j = 0; j < size; j++) {
      data.push(i === j ? { re: 1, im: 0 } : { re: 0, im: 0 });
    }
  }
  return { rows: size, cols: size, data };
}

/**
 * Create zero matrix
 */
export function matrixZeros(rows: number, cols: number): Matrix {
  const data = Array(rows * cols).fill({ re: 0, im: 0 });
  return { rows, cols, data };
}

export function matrixFromGenerator(size: number, generator: (row: number, col: number) => Complex): Matrix {
  const data: Complex[] = [];
  for (let i = 0; i < size; i++) {
    for (let j = 0; j < size; j++) {
      data.push(generator(i, j));
    }
  }
  return { rows: size, cols: size, data };
}

export function scaleMatrix(m: Matrix, factor: number): Matrix {
  return {
    rows: m.rows,
    cols: m.cols,
    data: m.data.map(value => scale(factor, value))
  };
}

export function addMatrices(a: Matrix, b: Matrix): Matrix {
  if (a.rows !== b.rows || a.cols !== b.cols) {
    throw new Error('Matrix dimensions must match for addition');
  }
  const data: Complex[] = [];
  for (let i = 0; i < a.data.length; i++) {
    data.push(add(a.data[i], b.data[i]));
  }
  return { rows: a.rows, cols: a.cols, data };
}

/**
 * Matrix multiplication
 */
export function matrixMultiply(a: Matrix, b: Matrix): Matrix {
  if (a.cols !== b.rows) {
    throw new Error(`Matrix dimensions mismatch: ${a.rows}x${a.cols} * ${b.rows}x${b.cols}`);
  }

  const result: Matrix = {
    rows: a.rows,
    cols: b.cols,
    data: []
  };

  for (let i = 0; i < a.rows; i++) {
    for (let j = 0; j < b.cols; j++) {
      let sum: Complex = { re: 0, im: 0 };
      for (let k = 0; k < a.cols; k++) {
        const aElement = a.data[i * a.cols + k];
        const bElement = b.data[k * b.cols + j];
        sum = add(sum, multiply(aElement, bElement));
      }
      result.data.push(sum);
    }
  }

  return result;
}

/**
 * Matrix conjugate transpose (Hermitian transpose)
 */
export function matrixConjugateTranspose(m: Matrix): Matrix {
  const result: Matrix = {
    rows: m.cols,
    cols: m.rows,
    data: []
  };

  for (let i = 0; i < m.cols; i++) {
    for (let j = 0; j < m.rows; j++) {
      result.data.push(conj(m.data[j * m.cols + i]));
    }
  }

  return result;
}

/**
 * Matrix norm (Frobenius norm of difference between two matrices)
 */
export function matrixNorm(a: Matrix, b: Matrix): number {
  if (a.rows !== b.rows || a.cols !== b.cols) {
    throw new Error('Matrix dimensions must match for norm calculation');
  }

  let sum = 0;
  for (let i = 0; i < a.data.length; i++) {
    const diff = subtract(a.data[i], b.data[i]);
    sum += abs(diff) ** 2;
  }

  return Math.sqrt(sum);
}

/**
 * QR Decomposition using Gram-Schmidt orthogonalization
 * Returns Q (orthonormal matrix)
 */
export function qrDecomposition(m: Matrix): Matrix {
  const n = m.rows;
  const Q: Matrix = {
    rows: n,
    cols: n,
    data: [...m.data]
  };

  // Gram-Schmidt orthogonalization
  for (let j = 0; j < n; j++) {
    // Orthogonalize column j against all previous columns
    for (let i = 0; i < j; i++) {
      // Compute projection: <Q[i], Q[j]>
      let dotProduct: Complex = { re: 0, im: 0 };
      for (let k = 0; k < n; k++) {
        const qi = Q.data[k * n + i];
        const qj = Q.data[k * n + j];
        dotProduct = add(dotProduct, multiply(conj(qi), qj));
      }

      // Subtract projection: Q[j] -= <Q[i], Q[j]> * Q[i]
      for (let k = 0; k < n; k++) {
        const qi = Q.data[k * n + i];
        const scaled = multiply(dotProduct, qi);
        Q.data[k * n + j] = subtract(Q.data[k * n + j], scaled);
      }
    }

    // Normalize column j
    let norm = 0;
    for (let k = 0; k < n; k++) {
      norm += abs(Q.data[k * n + j]) ** 2;
    }
    norm = Math.sqrt(norm);

    if (norm > 1e-15) {
      for (let k = 0; k < n; k++) {
        Q.data[k * n + j] = divide(Q.data[k * n + j], { re: norm, im: 0 });
      }
    }
  }

  return Q;
}

/**
 * Get column vector from matrix
 */
export function getColumn(m: Matrix, col: number): Complex[] {
  const result: Complex[] = [];
  for (let i = 0; i < m.rows; i++) {
    result.push(m.data[i * m.cols + col]);
  }
  return result;
}

/**
 * Get row vector from matrix
 */
export function getRow(m: Matrix, row: number): Complex[] {
  return m.data.slice(row * m.cols, (row + 1) * m.cols);
}

/**
 * Set column vector in matrix
 */
export function setColumn(m: Matrix, col: number, values: Complex[]): void {
  if (values.length !== m.rows) {
    throw new Error('Column length must match matrix rows');
  }
  for (let i = 0; i < m.rows; i++) {
    m.data[i * m.cols + col] = values[i];
  }
}

/**
 * Matrix-vector multiplication
 */
export function matrixVectorMultiply(m: Matrix, v: Complex[]): Complex[] {
  if (m.cols !== v.length) {
    throw new Error(`Matrix columns ${m.cols} must match vector length ${v.length}`);
  }

  const result: Complex[] = [];
  for (let i = 0; i < m.rows; i++) {
    let sum: Complex = { re: 0, im: 0 };
    for (let j = 0; j < m.cols; j++) {
      sum = add(sum, multiply(m.data[i * m.cols + j], v[j]));
    }
    result.push(sum);
  }
  return result;
}

/**
 * Tensor product (Kronecker product) of two matrices
 */
export function tensorProduct(a: Matrix, b: Matrix): Matrix {
  const result: Matrix = {
    rows: a.rows * b.rows,
    cols: a.cols * b.cols,
    data: []
  };

  for (let i = 0; i < a.rows; i++) {
    for (let k = 0; k < b.rows; k++) {
      for (let j = 0; j < a.cols; j++) {
        for (let l = 0; l < b.cols; l++) {
          const aElement = a.data[i * a.cols + j];
          const bElement = b.data[k * b.cols + l];
          result.data.push(multiply(aElement, bElement));
        }
      }
    }
  }

  return result;
}
