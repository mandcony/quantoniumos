/**
 * Quantum Simulator - QuantoniumOS Mobile
 *
 * Classical simulation of quantum circuits
 */

import { Complex, add, multiply, abs, conj } from '../rft/Complex';
import { Matrix, matrixVectorMultiply } from '../rft/Matrix';
import { QuantumGate } from './QuantumGates';

export interface QuantumState {
  numQubits: number;
  amplitudes: Complex[];
}

export interface MeasurementResult {
  outcome: string;
  probability: number;
}

/**
 * Quantum Circuit Simulator
 */
export class QuantumSimulator {
  private numQubits: number;
  private state: Complex[];

  constructor(numQubits: number) {
    this.numQubits = numQubits;
    const stateSize = Math.pow(2, numQubits);

    // Initialize to |0...0⟩ state
    this.state = new Array(stateSize).fill({ re: 0, im: 0 });
    this.state[0] = { re: 1, im: 0 };
  }

  /**
   * Get current quantum state
   */
  getState(): QuantumState {
    return {
      numQubits: this.numQubits,
      amplitudes: [...this.state]
    };
  }

  /**
   * Set state directly (for preparing specific states)
   */
  setState(amplitudes: Complex[]): void {
    if (amplitudes.length !== this.state.length) {
      throw new Error(`State size must be ${this.state.length}`);
    }

    // Normalize the state
    let norm = 0;
    for (const amp of amplitudes) {
      norm += abs(amp) ** 2;
    }
    norm = Math.sqrt(norm);

    this.state = amplitudes.map(amp => ({
      re: amp.re / norm,
      im: amp.im / norm
    }));
  }

  /**
   * Apply a quantum gate to specific qubits
   */
  applyGate(gate: QuantumGate, targetQubits: number[]): void {
    const gateSize = Math.log2(gate.size);
    if (targetQubits.length !== gateSize) {
      throw new Error(`Gate requires ${gateSize} qubits, got ${targetQubits.length}`);
    }

    // For simplicity, only support single and two-qubit gates
    if (gateSize === 1) {
      this.applySingleQubitGate(gate, targetQubits[0]);
    } else if (gateSize === 2) {
      this.applyTwoQubitGate(gate, targetQubits[0], targetQubits[1]);
    } else {
      throw new Error('Only single and two-qubit gates are supported');
    }
  }

  /**
   * Apply single-qubit gate
   */
  private applySingleQubitGate(gate: QuantumGate, targetQubit: number): void {
    const newState = [...this.state];
    const stateSize = this.state.length;
    const targetMask = 1 << targetQubit;

    for (let i = 0; i < stateSize; i++) {
      if ((i & targetMask) === 0) {
        const i0 = i;
        const i1 = i | targetMask;

        const amp0 = this.state[i0];
        const amp1 = this.state[i1];

        // Apply 2x2 gate matrix
        const g00 = gate.matrix.data[0];
        const g01 = gate.matrix.data[1];
        const g10 = gate.matrix.data[2];
        const g11 = gate.matrix.data[3];

        newState[i0] = add(multiply(g00, amp0), multiply(g01, amp1));
        newState[i1] = add(multiply(g10, amp0), multiply(g11, amp1));
      }
    }

    this.state = newState;
  }

  /**
   * Apply two-qubit gate
   */
  private applyTwoQubitGate(gate: QuantumGate, control: number, target: number): void {
    const newState = [...this.state];
    const stateSize = this.state.length;
    const controlMask = 1 << control;
    const targetMask = 1 << target;

    for (let i = 0; i < stateSize; i++) {
      if ((i & controlMask) === 0 && (i & targetMask) === 0) {
        const i00 = i;
        const i01 = i | targetMask;
        const i10 = i | controlMask;
        const i11 = i | controlMask | targetMask;

        const amp00 = this.state[i00];
        const amp01 = this.state[i01];
        const amp10 = this.state[i10];
        const amp11 = this.state[i11];

        // Apply 4x4 gate matrix
        for (let row = 0; row < 4; row++) {
          let sum: Complex = { re: 0, im: 0 };

          sum = add(sum, multiply(gate.matrix.data[row * 4 + 0], amp00));
          sum = add(sum, multiply(gate.matrix.data[row * 4 + 1], amp01));
          sum = add(sum, multiply(gate.matrix.data[row * 4 + 2], amp10));
          sum = add(sum, multiply(gate.matrix.data[row * 4 + 3], amp11));

          const indices = [i00, i01, i10, i11];
          newState[indices[row]] = sum;
        }
      }
    }

    this.state = newState;
  }

  /**
   * Measure a specific qubit
   */
  measureQubit(qubit: number): { outcome: number; probability: number } {
    if (qubit >= this.numQubits) {
      throw new Error(`Qubit ${qubit} does not exist`);
    }

    const mask = 1 << qubit;
    let prob0 = 0;
    let prob1 = 0;

    // Calculate probabilities
    for (let i = 0; i < this.state.length; i++) {
      const prob = abs(this.state[i]) ** 2;
      if ((i & mask) === 0) {
        prob0 += prob;
      } else {
        prob1 += prob;
      }
    }

    // Simulate measurement (randomly choose based on probabilities)
    const outcome = Math.random() < prob0 ? 0 : 1;
    const probability = outcome === 0 ? prob0 : prob1;

    // Collapse state
    const norm = Math.sqrt(probability);
    for (let i = 0; i < this.state.length; i++) {
      if (((i & mask) === 0 && outcome === 0) || ((i & mask) !== 0 && outcome === 1)) {
        this.state[i] = {
          re: this.state[i].re / norm,
          im: this.state[i].im / norm
        };
      } else {
        this.state[i] = { re: 0, im: 0 };
      }
    }

    return { outcome, probability };
  }

  /**
   * Measure all qubits (computational basis)
   */
  measureAll(): MeasurementResult {
    const probabilities: number[] = [];

    for (let i = 0; i < this.state.length; i++) {
      probabilities[i] = abs(this.state[i]) ** 2;
    }

    // Choose outcome based on probabilities
    const rand = Math.random();
    let cumulative = 0;
    let outcome = 0;

    for (let i = 0; i < probabilities.length; i++) {
      cumulative += probabilities[i];
      if (rand < cumulative) {
        outcome = i;
        break;
      }
    }

    // Convert to binary string
    const binaryString = outcome.toString(2).padStart(this.numQubits, '0');

    return {
      outcome: binaryString,
      probability: probabilities[outcome]
    };
  }

  /**
   * Get probability distribution
   */
  getProbabilities(): { state: string; probability: number }[] {
    const results: { state: string; probability: number }[] = [];

    for (let i = 0; i < this.state.length; i++) {
      const prob = abs(this.state[i]) ** 2;
      if (prob > 1e-10) {
        const binaryString = i.toString(2).padStart(this.numQubits, '0');
        results.push({
          state: `|${binaryString}⟩`,
          probability: prob
        });
      }
    }

    return results.sort((a, b) => b.probability - a.probability);
  }

  /**
   * Calculate fidelity with target state
   */
  fidelity(targetState: Complex[]): number {
    if (targetState.length !== this.state.length) {
      throw new Error('State dimensions must match');
    }

    let innerProduct: Complex = { re: 0, im: 0 };
    for (let i = 0; i < this.state.length; i++) {
      innerProduct = add(innerProduct, multiply(conj(targetState[i]), this.state[i]));
    }

    return abs(innerProduct) ** 2;
  }

  /**
   * Reset to |0...0⟩ state
   */
  reset(): void {
    this.state = new Array(this.state.length).fill({ re: 0, im: 0 });
    this.state[0] = { re: 1, im: 0 };
  }
}

/**
 * Pre-built quantum circuits
 */
export class QuantumCircuits {
  /**
   * Create Bell state circuit
   */
  static bellState(): QuantumSimulator {
    const sim = new QuantumSimulator(2);

    // Import gates inline to avoid circular dependency
    const H: Matrix = {
      rows: 2,
      cols: 2,
      data: [
        { re: 1/Math.sqrt(2), im: 0 }, { re: 1/Math.sqrt(2), im: 0 },
        { re: 1/Math.sqrt(2), im: 0 }, { re: -1/Math.sqrt(2), im: 0 }
      ]
    };

    const CNOT: Matrix = {
      rows: 4,
      cols: 4,
      data: [
        { re: 1, im: 0 }, { re: 0, im: 0 }, { re: 0, im: 0 }, { re: 0, im: 0 },
        { re: 0, im: 0 }, { re: 1, im: 0 }, { re: 0, im: 0 }, { re: 0, im: 0 },
        { re: 0, im: 0 }, { re: 0, im: 0 }, { re: 0, im: 0 }, { re: 1, im: 0 },
        { re: 0, im: 0 }, { re: 0, im: 0 }, { re: 1, im: 0 }, { re: 0, im: 0 }
      ]
    };

    sim.applyGate(new QuantumGate(H, 'H'), [0]);
    sim.applyGate(new QuantumGate(CNOT, 'CNOT'), [0, 1]);

    return sim;
  }

  /**
   * Create GHZ state circuit (generalized Bell state for n qubits)
   */
  static ghzState(numQubits: number): QuantumSimulator {
    if (numQubits < 2) {
      throw new Error('GHZ state requires at least 2 qubits');
    }

    const sim = new QuantumSimulator(numQubits);

    const H: Matrix = {
      rows: 2,
      cols: 2,
      data: [
        { re: 1/Math.sqrt(2), im: 0 }, { re: 1/Math.sqrt(2), im: 0 },
        { re: 1/Math.sqrt(2), im: 0 }, { re: -1/Math.sqrt(2), im: 0 }
      ]
    };

    const CNOT: Matrix = {
      rows: 4,
      cols: 4,
      data: [
        { re: 1, im: 0 }, { re: 0, im: 0 }, { re: 0, im: 0 }, { re: 0, im: 0 },
        { re: 0, im: 0 }, { re: 1, im: 0 }, { re: 0, im: 0 }, { re: 0, im: 0 },
        { re: 0, im: 0 }, { re: 0, im: 0 }, { re: 0, im: 0 }, { re: 1, im: 0 },
        { re: 0, im: 0 }, { re: 0, im: 0 }, { re: 1, im: 0 }, { re: 0, im: 0 }
      ]
    };

    // Apply Hadamard to first qubit
    sim.applyGate(new QuantumGate(H, 'H'), [0]);

    // Apply CNOT gates to entangle all qubits
    for (let i = 1; i < numQubits; i++) {
      sim.applyGate(new QuantumGate(CNOT, 'CNOT'), [0, i]);
    }

    return sim;
  }
}
