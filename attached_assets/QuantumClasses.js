// QuantumClasses.js

export class Complex {
  constructor(real, imag) {
    this.real = real;
    this.imag = imag;
  }
  add(other) {
    return new Complex(this.real + other.real, this.imag + other.imag);
  }
  multiply(other) {
    const re = this.real * other.real - this.imag * other.imag;
    const im = this.real * other.imag + this.imag * other.real;
    return new Complex(re, im);
  }
  toString() {
    const sign = this.imag >= 0 ? '+' : '';
    return `${this.real}${sign}${this.imag}i`;
  }
}

export class SymbolicAmplitude {
  constructor(expr) {
    this.expr = expr;
  }
  static fromString(s) {
    return new SymbolicAmplitude(s);
  }
  toString() {
    return this.expr;
  }
  add(other) {
    return new SymbolicAmplitude(`(${this.expr} + ${other.expr})`);
  }
  multiply(other) {
    return new SymbolicAmplitude(`(${this.expr} * ${other.expr})`);
  }
}

export class MultiQubitState {
  constructor(numQubits) {
    this.numQubits = numQubits;
    this.size = 1 << numQubits; // 2^numQubits
    this.stateVector = new Array(this.size).fill(null)
      .map((_, i) => SymbolicAmplitude.fromString(`|${i}>`));
  }

  applySingleQubitGate(gateMatrix, target) {
    const newVec = [...this.stateVector];
    const blockSize = 1 << target;
    const doubleBlock = blockSize << 1;

    for (let start = 0; start < this.size; start += doubleBlock) {
      for (let offset = 0; offset < blockSize; offset++) {
        const i0 = start + offset;
        const i1 = i0 + blockSize;
        const amp0 = this.stateVector[i0];
        const amp1 = this.stateVector[i1];

        const m00 = gateMatrix[0][0](amp0);
        const m01 = gateMatrix[0][1](amp1);
        const m10 = gateMatrix[1][0](amp0);
        const m11 = gateMatrix[1][1](amp1);

        newVec[i0] = SymbolicAmplitude.fromString(`(${m00.toString()} + ${m01.toString()})`);
        newVec[i1] = SymbolicAmplitude.fromString(`(${m10.toString()} + ${m11.toString()})`);
      }
    }
    this.stateVector = newVec;
  }

  applyTwoQubitGate(gateName, control, target) {
    if (gateName !== 'CNOT') {
      console.warn(`Only 'CNOT' is implemented. Attempted: ${gateName}`);
      return;
    }
    const newVec = [...this.stateVector];
    for (let i = 0; i < this.size; i++) {
      newVec[i] = this.stateVector[i];
    }
    // Basic CNOT
    for (let i = 0; i < this.size; i++) {
      const controlBit = (i >> control) & 1;
      if (controlBit === 1) {
        const flipped = i ^ (1 << target);
        newVec[flipped] = this.stateVector[i];
      }
    }
    this.stateVector = newVec;
  }

  measureAll() {
    const idx = Math.floor(Math.random() * this.size);
    const collapsed = new Array(this.size).fill(SymbolicAmplitude.fromString('0'));
    collapsed[idx] = SymbolicAmplitude.fromString(`MEASURED-${idx}`);
    this.stateVector = collapsed;
    return idx;
  }

  getAmplitudeInfo() {
    return this.stateVector.map(s => s.toString());
  }
}
