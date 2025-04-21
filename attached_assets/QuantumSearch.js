// services/QuantumSearch.js
import * as math from 'mathjs';

class QuantumSearch {
  constructor() {
    this.groverIterations = 0;
  }

  createInitialState(numQubits) {
    const state = math.zeros(1 << numQubits)._data;
    state[0] = 1; // |0...0>
    return state;
  }

  createHadamardGate(numQubits) {
    // 1-qubit H
    const h1 = math.matrix([[1, 1], [1, -1]]).map((val) => val / Math.sqrt(2));
    // tensor repeatedly
    let combined = math.matrix([[1]]);
    for (let i = 0; i < numQubits; i++) {
      combined = math.kron(combined, h1);
    }
    return combined;
  }

  createPhaseInversionGate(containers, targetFrequency, numQubits) {
    const size = 1 << numQubits;
    const gate = math.identity(size)._data;

    // Mark items that match resonance => multiply them by -1
    for (let i = 0; i < containers.length; i++) {
      const container = containers[i];
      if (container.checkResonance(targetFrequency)) {
        // The index is i in normal integer, if we assume 1-to-1 mapping
        if (i < size) {
          gate[i][i] = -1;
        }
      }
    }
    return math.matrix(gate);
  }

  createDiffusionOperator(numQubits) {
    const size = 1 << numQubits;
    const hadamard = this.createHadamardGate(numQubits);
    const identity = math.identity(size).clone();
    const inversion = math.clone(identity);

    // Invert about the average
    for (let i = 0; i < size; i++) {
      inversion._data[i][i] = -1;
    }
    // Fix the [0][0] position to +1 (depending on the variant used)
    inversion._data[0][0] = 1;

    // D = H * inversion * H
    const diffusionMatrix = math.multiply(
      math.multiply(hadamard, inversion),
      hadamard
    );
    return diffusionMatrix;
  }

  search(containers, targetFrequency, numQubitsOverride = null, resonanceThreshold = 0.1) {
    if (!containers || containers.length === 0) {
      console.warn("QuantumSearch: Empty container list. Returning null.");
      return null;
    }

    let numQubits = numQubitsOverride || Math.ceil(Math.log2(containers.length));
    if (containers.length > (1 << numQubits)) {
      numQubits = Math.ceil(Math.log2(containers.length));
    }

    // if there's only 1 container
    if (containers.length === 1) {
      if (containers[0].checkResonance(targetFrequency, resonanceThreshold)) {
        return containers[0];
      }
      return null;
    }

    // Guard against too big
    const maxQubits = 10;
    if (numQubits > maxQubits) {
      throw new Error(`QuantumSearch: Number of qubits (${numQubits}) exceeds max allowed (${maxQubits}).`);
    }

    let stateVector = this.createInitialState(numQubits);
    const hadamardGate = this.createHadamardGate(numQubits);

    // initial apply H to all qubits
    stateVector = math.multiply(hadamardGate, stateVector);

    // approximate # of Grover iterations
    const numIterations = Math.floor(Math.sqrt(Math.pow(2, numQubits)));

    for (let i = 0; i < numIterations; i++) {
      const phaseInversionGate = this.createPhaseInversionGate(containers, targetFrequency, numQubits);
      const diffusionOperator = this.createDiffusionOperator(numQubits);

      // mark
      stateVector = math.multiply(phaseInversionGate, stateVector);
      // diffuse
      stateVector = math.multiply(diffusionOperator, stateVector);
    }

    // find index with max amplitude
    let maxAmplitude = 0;
    let maxIndex = -1;
    for (let i = 0; i < stateVector.length; i++) {
      const amplitude = Math.abs(stateVector[i]);
      if (amplitude > maxAmplitude) {
        maxAmplitude = amplitude;
        maxIndex = i;
      }
    }
    if (maxIndex < 0) return null;

    if (maxIndex < containers.length) {
      const selected = containers[maxIndex];
      // double-check resonance
      if (selected.checkResonance(targetFrequency, resonanceThreshold)) {
        return selected;
      }
    }
    return null;
  }

  getGroverIterations() {
    return this.groverIterations;
  }
}

export default QuantumSearch;
