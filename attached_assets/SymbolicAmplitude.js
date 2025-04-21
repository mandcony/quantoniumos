// services/SymbolicAmplitude.js

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
