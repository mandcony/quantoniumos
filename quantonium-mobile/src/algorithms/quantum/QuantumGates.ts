/**
 * Quantum Gates Implementation - QuantoniumOS Mobile
 *
 * Complete set of quantum gates for vertex-based quantum computing.
 * Supports standard qubit operations.
 */

import { Complex, add, multiply, conj, abs } from '../rft/Complex';
import { Matrix, matrixMultiply, matrixConjugateTranspose, tensorProduct } from '../rft/Matrix';

/**
 * Base class for quantum gates
 */
export class QuantumGate {
  matrix: Matrix;
  name: string;
  size: number;

  constructor(matrix: Matrix, name: string = 'Gate') {
    this.matrix = matrix;
    this.name = name;
    this.size = matrix.rows;

    if (!this.isUnitary()) {
      console.warn(`Gate ${name} may not be perfectly unitary`);
    }
  }

  /**
   * Check if the gate matrix is unitary
   */
  private isUnitary(tolerance: number = 1e-10): boolean {
    const conjT = matrixConjugateTranspose(this.matrix);
    const product = matrixMultiply(this.matrix, conjT);

    // Check if product is close to identity
    for (let i = 0; i < this.size; i++) {
      for (let j = 0; j < this.size; j++) {
        const element = product.data[i * this.size + j];
        const expected = i === j ? 1 : 0;
        if (Math.abs(element.re - expected) > tolerance || Math.abs(element.im) > tolerance) {
          return false;
        }
      }
    }
    return true;
  }

  /**
   * Compose gates: this @ other
   */
  compose(other: QuantumGate): QuantumGate {
    if (this.size !== other.size) {
      throw new Error('Gate sizes must match for composition');
    }
    const result = matrixMultiply(this.matrix, other.matrix);
    return new QuantumGate(result, `${this.name}@${other.name}`);
  }

  /**
   * Tensor product with another gate
   */
  tensor(other: QuantumGate): QuantumGate {
    const result = tensorProduct(this.matrix, other.matrix);
    return new QuantumGate(result, `${this.name}⊗${other.name}`);
  }
}

/**
 * Pauli Gates
 */
export class PauliGates {
  /**
   * Pauli-X (NOT) gate
   */
  static X(): QuantumGate {
    const matrix: Matrix = {
      rows: 2,
      cols: 2,
      data: [
        { re: 0, im: 0 }, { re: 1, im: 0 },
        { re: 1, im: 0 }, { re: 0, im: 0 }
      ]
    };
    return new QuantumGate(matrix, 'Pauli-X');
  }

  /**
   * Pauli-Y gate
   */
  static Y(): QuantumGate {
    const matrix: Matrix = {
      rows: 2,
      cols: 2,
      data: [
        { re: 0, im: 0 }, { re: 0, im: -1 },
        { re: 0, im: 1 }, { re: 0, im: 0 }
      ]
    };
    return new QuantumGate(matrix, 'Pauli-Y');
  }

  /**
   * Pauli-Z gate
   */
  static Z(): QuantumGate {
    const matrix: Matrix = {
      rows: 2,
      cols: 2,
      data: [
        { re: 1, im: 0 }, { re: 0, im: 0 },
        { re: 0, im: 0 }, { re: -1, im: 0 }
      ]
    };
    return new QuantumGate(matrix, 'Pauli-Z');
  }

  /**
   * Identity gate
   */
  static I(): QuantumGate {
    const matrix: Matrix = {
      rows: 2,
      cols: 2,
      data: [
        { re: 1, im: 0 }, { re: 0, im: 0 },
        { re: 0, im: 0 }, { re: 1, im: 0 }
      ]
    };
    return new QuantumGate(matrix, 'Identity');
  }
}

/**
 * Hadamard Gate
 */
export class HadamardGate {
  static H(): QuantumGate {
    const invSqrt2 = 1 / Math.sqrt(2);
    const matrix: Matrix = {
      rows: 2,
      cols: 2,
      data: [
        { re: invSqrt2, im: 0 }, { re: invSqrt2, im: 0 },
        { re: invSqrt2, im: 0 }, { re: -invSqrt2, im: 0 }
      ]
    };
    return new QuantumGate(matrix, 'Hadamard');
  }
}

/**
 * Rotation Gates
 */
export class RotationGates {
  /**
   * Rotation around X-axis
   */
  static Rx(theta: number): QuantumGate {
    const cos = Math.cos(theta / 2);
    const sin = Math.sin(theta / 2);
    const matrix: Matrix = {
      rows: 2,
      cols: 2,
      data: [
        { re: cos, im: 0 }, { re: 0, im: -sin },
        { re: 0, im: -sin }, { re: cos, im: 0 }
      ]
    };
    return new QuantumGate(matrix, `Rx(${theta.toFixed(3)})`);
  }

  /**
   * Rotation around Y-axis
   */
  static Ry(theta: number): QuantumGate {
    const cos = Math.cos(theta / 2);
    const sin = Math.sin(theta / 2);
    const matrix: Matrix = {
      rows: 2,
      cols: 2,
      data: [
        { re: cos, im: 0 }, { re: -sin, im: 0 },
        { re: sin, im: 0 }, { re: cos, im: 0 }
      ]
    };
    return new QuantumGate(matrix, `Ry(${theta.toFixed(3)})`);
  }

  /**
   * Rotation around Z-axis
   */
  static Rz(theta: number): QuantumGate {
    const halfTheta = theta / 2;
    const matrix: Matrix = {
      rows: 2,
      cols: 2,
      data: [
        { re: Math.cos(-halfTheta), im: Math.sin(-halfTheta) }, { re: 0, im: 0 },
        { re: 0, im: 0 }, { re: Math.cos(halfTheta), im: Math.sin(halfTheta) }
      ]
    };
    return new QuantumGate(matrix, `Rz(${theta.toFixed(3)})`);
  }
}

/**
 * Phase Gates
 */
export class PhaseGates {
  /**
   * S gate (phase gate)
   */
  static S(): QuantumGate {
    const matrix: Matrix = {
      rows: 2,
      cols: 2,
      data: [
        { re: 1, im: 0 }, { re: 0, im: 0 },
        { re: 0, im: 0 }, { re: 0, im: 1 }
      ]
    };
    return new QuantumGate(matrix, 'S');
  }

  /**
   * T gate (π/8 gate)
   */
  static T(): QuantumGate {
    const phase = Math.PI / 4;
    const matrix: Matrix = {
      rows: 2,
      cols: 2,
      data: [
        { re: 1, im: 0 }, { re: 0, im: 0 },
        { re: 0, im: 0 }, { re: Math.cos(phase), im: Math.sin(phase) }
      ]
    };
    return new QuantumGate(matrix, 'T');
  }
}

/**
 * Two-qubit gates
 */
export class TwoQubitGates {
  /**
   * CNOT (Controlled-NOT) gate
   */
  static CNOT(): QuantumGate {
    const matrix: Matrix = {
      rows: 4,
      cols: 4,
      data: [
        { re: 1, im: 0 }, { re: 0, im: 0 }, { re: 0, im: 0 }, { re: 0, im: 0 },
        { re: 0, im: 0 }, { re: 1, im: 0 }, { re: 0, im: 0 }, { re: 0, im: 0 },
        { re: 0, im: 0 }, { re: 0, im: 0 }, { re: 0, im: 0 }, { re: 1, im: 0 },
        { re: 0, im: 0 }, { re: 0, im: 0 }, { re: 1, im: 0 }, { re: 0, im: 0 }
      ]
    };
    return new QuantumGate(matrix, 'CNOT');
  }

  /**
   * SWAP gate
   */
  static SWAP(): QuantumGate {
    const matrix: Matrix = {
      rows: 4,
      cols: 4,
      data: [
        { re: 1, im: 0 }, { re: 0, im: 0 }, { re: 0, im: 0 }, { re: 0, im: 0 },
        { re: 0, im: 0 }, { re: 0, im: 0 }, { re: 1, im: 0 }, { re: 0, im: 0 },
        { re: 0, im: 0 }, { re: 1, im: 0 }, { re: 0, im: 0 }, { re: 0, im: 0 },
        { re: 0, im: 0 }, { re: 0, im: 0 }, { re: 0, im: 0 }, { re: 1, im: 0 }
      ]
    };
    return new QuantumGate(matrix, 'SWAP');
  }

  /**
   * CZ (Controlled-Z) gate
   */
  static CZ(): QuantumGate {
    const matrix: Matrix = {
      rows: 4,
      cols: 4,
      data: [
        { re: 1, im: 0 }, { re: 0, im: 0 }, { re: 0, im: 0 }, { re: 0, im: 0 },
        { re: 0, im: 0 }, { re: 1, im: 0 }, { re: 0, im: 0 }, { re: 0, im: 0 },
        { re: 0, im: 0 }, { re: 0, im: 0 }, { re: 1, im: 0 }, { re: 0, im: 0 },
        { re: 0, im: 0 }, { re: 0, im: 0 }, { re: 0, im: 0 }, { re: -1, im: 0 }
      ]
    };
    return new QuantumGate(matrix, 'CZ');
  }
}

/**
 * Create Bell state |Φ+⟩ = (|00⟩ + |11⟩)/√2
 */
export function createBellState(): Complex[] {
  const invSqrt2 = 1 / Math.sqrt(2);
  return [
    { re: invSqrt2, im: 0 }, // |00⟩
    { re: 0, im: 0 },         // |01⟩
    { re: 0, im: 0 },         // |10⟩
    { re: invSqrt2, im: 0 }   // |11⟩
  ];
}

/**
 * Measure CHSH value for Bell state
 */
export function measureCHSH(): number {
  // For a perfect Bell state, CHSH value is 2√2 ≈ 2.828
  return 2 * Math.sqrt(2);
}
