// tests/multiQubitState.test.js
import { MultiQubitState } from '../services/multiQubitState';
import { SymbolicAmplitude } from '../services/SymbolicAmplitude';

describe('MultiQubitState Class', () => {
  it('should initialize the state vector with symbolic amplitudes', () => {
    const numQubits = 3;
    const state = new MultiQubitState(numQubits);
    expect(state.size).toBe(8);
    expect(state.stateVector[0].toString()).toBe("|0>");
    expect(state.stateVector[7].toString()).toBe("|7>");
  });

  it('should apply a single qubit gate (simple test)', () => {
    const state = new MultiQubitState(2);
    // trivial X gate as a sample
    const xGate = [
      [
        (amp) => SymbolicAmplitude.fromString(`0*${amp.expr}`),
        (amp) => SymbolicAmplitude.fromString(`1*${amp.expr}`)
      ],
      [
        (amp) => SymbolicAmplitude.fromString(`1*${amp.expr}`),
        (amp) => SymbolicAmplitude.fromString(`0*${amp.expr}`)
      ],
    ];
    state.applySingleQubitGate(xGate, 0);
    // Just ensure we got new expressions in the first half
    expect(state.stateVector[0].toString()).toBe("(0*|0> + 1*|1>)");
  });

  it('should measure a random index', () => {
    const state = new MultiQubitState(2);
    const idx = state.measureAll();
    expect(idx).toBeGreaterThanOrEqual(0);
    expect(idx).toBeLessThan(4); // size = 4
    // check collapsed
    const nonZero = state.stateVector.filter(x => x.expr.startsWith("MEASURED"));
    expect(nonZero.length).toBe(1);
  });
});
