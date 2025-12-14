/**
 * Quantum Simulator Screen
 * Interactive quantum circuit simulator
 */

import React, { useState } from 'react';
import {
  View,
  Text,
  StyleSheet,
  TouchableOpacity,
  ScrollView,
} from 'react-native';
import ScreenShell from '../components/ScreenShell';
import { QuantumSimulator } from '../algorithms/quantum/QuantumSimulator';
import {
  PauliGates,
  HadamardGate,
  TwoQubitGates,
  measureCHSH,
} from '../algorithms/quantum/QuantumGates';
import {
  borderRadius,
  colors,
  shadows,
  spacing,
  typography,
} from '../constants/DesignSystem';

export default function QuantumSimulatorScreen() {
  const [simulator, setSimulator] = useState(() => new QuantumSimulator(2));
  const [output, setOutput] = useState<string>('Quantum Simulator initialized with 2 qubits in |00⟩ state');

  const appendOutput = (text: string) => {
    setOutput(prev => prev + '\n\n' + text);
  };

  const showProbabilities = () => {
    const probs = simulator.getProbabilities();
    const lines = probs.map((p) => `${p.state}: ${(p.probability * 100).toFixed(2)}%`);
    appendOutput('State probabilities:\n' + (lines.length ? lines.join('\n') : 'All zero amplitudes'));
  };

  const applyHadamard = () => {
    simulator.applyGate(HadamardGate.H(), [0]);
    appendOutput('Applied Hadamard gate to qubit 0');
    showProbabilities();
  };

  const applyPauliX = () => {
    simulator.applyGate(PauliGates.X(), [0]);
    appendOutput('Applied Pauli-X (NOT) gate to qubit 0');
    showProbabilities();
  };

  const applyCNOT = () => {
    simulator.applyGate(TwoQubitGates.CNOT(), [0, 1]);
    appendOutput('Applied CNOT gate (control=0, target=1)');
    showProbabilities();
  };

  const createBell = () => {
    const newSim = new QuantumSimulator(2);
    newSim.applyGate(HadamardGate.H(), [0]);
    newSim.applyGate(TwoQubitGates.CNOT(), [0, 1]);
    setSimulator(newSim);
    appendOutput('Created Bell state |Φ+⟩ = (|00⟩ + |11⟩)/√2');
    showProbabilities();
  };

  const measure = () => {
    const result = simulator.measureAll();
    appendOutput(`Measured: |${result.outcome}⟩`);
    appendOutput('State collapsed. System reset to measured state.');
  };

  const reset = () => {
    const newSim = new QuantumSimulator(2);
    setSimulator(newSim);
    setOutput('Quantum Simulator reset to |00⟩ state');
  };

  return (
    <ScreenShell
      title="Quantum Simulator"
      subtitle="Two-qubit Φ-RFT aligned classical emulator"
    >
      <View style={styles.leadCopy}>
        <Text style={styles.leadText}>
          Interact with the QuantoniumOS circuit sandbox. Apply gates, generate entangled
          states, and inspect measurement probabilities using the same logic as the
          desktop simulator.
        </Text>
      </View>

      <View style={styles.sectionCard}>
        <Text style={styles.sectionTitle}>Single-Qubit Gates</Text>
        <View style={styles.buttonRow}>
          <TouchableOpacity style={styles.gateButton} onPress={applyHadamard}>
            <Text style={styles.gateButtonSymbol}>H</Text>
            <Text style={styles.gateButtonLabel}>Hadamard</Text>
          </TouchableOpacity>
          <TouchableOpacity style={styles.gateButton} onPress={applyPauliX}>
            <Text style={styles.gateButtonSymbol}>X</Text>
            <Text style={styles.gateButtonLabel}>Pauli-X</Text>
          </TouchableOpacity>
        </View>

        <Text style={styles.sectionTitle}>Two-Qubit Gates</Text>
        <View style={styles.buttonRow}>
          <TouchableOpacity style={styles.gateButton} onPress={applyCNOT}>
            <Text style={styles.gateButtonSymbol}>CNOT</Text>
            <Text style={styles.gateButtonLabel}>Controlled-NOT</Text>
          </TouchableOpacity>
        </View>

        <Text style={styles.sectionTitle}>Pre-Built Circuits</Text>
        <View style={styles.buttonRow}>
          <TouchableOpacity style={[styles.gateButton, styles.highlightButton]} onPress={createBell}>
            <Text style={styles.gateButtonSymbol}>Φ+</Text>
            <Text style={styles.gateButtonLabel}>Bell Entanglement</Text>
          </TouchableOpacity>
        </View>
      </View>

      <View style={styles.sectionCard}>
        <Text style={styles.sectionTitle}>Operations</Text>
        <View style={styles.buttonRow}>
          <TouchableOpacity style={[styles.actionButton, styles.measureButton]} onPress={measure}>
            <Text style={styles.actionButtonText}>Measure</Text>
          </TouchableOpacity>
          <TouchableOpacity style={[styles.actionButton, styles.resetButton]} onPress={reset}>
            <Text style={styles.actionButtonText}>Reset</Text>
          </TouchableOpacity>
        </View>
      </View>

      <View style={styles.logCard}>
        <Text style={styles.sectionTitle}>Simulation Output</Text>
        <Text style={styles.sectionSubtitle}>Logs stream in chronological order.</Text>
        <ScrollView style={styles.outputScroll} nestedScrollEnabled>
          <Text style={styles.outputText}>{output}</Text>
        </ScrollView>
      </View>
    </ScreenShell>
  );
}

const styles = StyleSheet.create({
  leadCopy: {
    marginBottom: spacing.xl,
  },
  leadText: {
    fontSize: typography.body,
    lineHeight: typography.body + 6,
    color: colors.darkGray,
  },
  sectionCard: {
    backgroundColor: colors.surfaceElevated,
    borderRadius: borderRadius.xl,
    padding: spacing.xl,
    marginBottom: spacing.xl,
    borderWidth: 1,
    borderColor: 'rgba(52, 152, 219, 0.18)',
    ...shadows.sm,
  },
  sectionTitle: {
    fontSize: typography.subtitle,
    color: colors.dark,
    fontWeight: '600',
    letterSpacing: 0.6,
    marginBottom: spacing.sm,
  },
  sectionSubtitle: {
    fontSize: typography.small,
    color: colors.gray,
    marginBottom: spacing.md,
  },
  buttonRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    marginBottom: spacing.md,
  },
  gateButton: {
    flex: 1,
    backgroundColor: colors.white,
    borderRadius: borderRadius.lg,
    paddingVertical: spacing.lg,
    marginHorizontal: spacing.sm,
    alignItems: 'center',
    borderWidth: 1,
    borderColor: 'rgba(52, 152, 219, 0.2)',
    ...shadows.sm,
  },
  gateButtonSymbol: {
    fontSize: typography.title,
    color: colors.primary,
    fontWeight: '600',
    letterSpacing: 1,
  },
  gateButtonLabel: {
    marginTop: spacing.xs,
    fontSize: typography.small,
    color: colors.gray,
  },
  highlightButton: {
    backgroundColor: colors.offWhite,
    borderColor: 'rgba(243, 196, 15, 0.45)',
  },
  actionButton: {
    flex: 1,
    paddingVertical: spacing.md,
    marginHorizontal: spacing.sm,
    alignItems: 'center',
    borderRadius: borderRadius.md,
    borderWidth: 1,
  },
  measureButton: {
    backgroundColor: colors.white,
    borderColor: 'rgba(46, 204, 113, 0.35)',
    ...shadows.sm,
  },
  resetButton: {
    backgroundColor: colors.white,
    borderColor: 'rgba(231, 76, 60, 0.35)',
    ...shadows.sm,
  },
  actionButtonText: {
    fontSize: typography.body,
    color: colors.dark,
    fontWeight: '600',
  },
  logCard: {
    backgroundColor: colors.surfaceElevated,
    borderRadius: borderRadius.xl,
    padding: spacing.xl,
    borderWidth: 1,
    borderColor: 'rgba(52, 152, 219, 0.18)',
    ...shadows.sm,
    minHeight: 260,
  },
  outputScroll: {
    marginTop: spacing.sm,
    maxHeight: 240,
  },
  outputText: {
    fontSize: typography.small,
    color: colors.dark,
    fontFamily: 'monospace',
    lineHeight: typography.small + 4,
  },
});
