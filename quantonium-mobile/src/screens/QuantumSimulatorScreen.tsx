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
import { LinearGradient } from 'expo-linear-gradient';
import { QuantumSimulator, QuantumCircuits } from '../algorithms/quantum/QuantumSimulator';
import { PauliGates, HadamardGate, TwoQubitGates, createBellState, measureCHSH } from '../algorithms/quantum/QuantumGates';

export default function QuantumSimulatorScreen() {
  const [simulator, setSimulator] = useState(() => new QuantumSimulator(2));
  const [output, setOutput] = useState<string>('Quantum Simulator initialized with 2 qubits in |00⟩ state');

  const appendOutput = (text: string) => {
    setOutput(prev => prev + '\n\n' + text);
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
    appendOutput('Created Bell State |Φ+⟩ = (|00⟩ + |11⟩)/√2');
    const probs = newSim.getProbabilities();
    const probStr = probs.map(p => `${p.state}: ${(p.probability * 100).toFixed(1)}%`).join('\n');
    appendOutput('Probabilities:\n' + probStr);

    const chsh = measureCHSH();
    appendOutput(`CHSH value: ${chsh.toFixed(3)} (violates classical limit of 2.0)`);
  };

  const measure = () => {
    const result = simulator.measureAll();
    appendOutput(`Measurement result: ${result.outcome} (probability: ${(result.probability * 100).toFixed(1)}%)`);
  };

  const showProbabilities = () => {
    const probs = simulator.getProbabilities();
    if (probs.length === 0) {
      appendOutput('State collapsed after measurement');
      return;
    }
    const probStr = probs.map(p => `${p.state}: ${(p.probability * 100).toFixed(1)}%`).join(', ');
    appendOutput('Current probabilities: ' + probStr);
  };

  const reset = () => {
    simulator.reset();
    setOutput('Simulator reset to |00⟩ state');
  };

  return (
    <LinearGradient colors={['#4facfe', '#00f2fe']} style={styles.container}>
      <ScrollView style={styles.scrollView} contentContainerStyle={styles.scrollContent}>
        <View style={styles.header}>
          <Text style={styles.headerTitle}>⚛️ Quantum Simulator</Text>
          <Text style={styles.headerSubtitle}>Classical Quantum Circuit Simulation</Text>
        </View>

        <View style={styles.controls}>
          <Text style={styles.sectionTitle}>Single-Qubit Gates</Text>
          <View style={styles.buttonRow}>
            <TouchableOpacity style={styles.gateButton} onPress={applyHadamard}>
              <Text style={styles.gateButtonText}>H</Text>
              <Text style={styles.gateLabel}>Hadamard</Text>
            </TouchableOpacity>
            <TouchableOpacity style={styles.gateButton} onPress={applyPauliX}>
              <Text style={styles.gateButtonText}>X</Text>
              <Text style={styles.gateLabel}>Pauli-X</Text>
            </TouchableOpacity>
          </View>

          <Text style={styles.sectionTitle}>Two-Qubit Gates</Text>
          <View style={styles.buttonRow}>
            <TouchableOpacity style={styles.gateButton} onPress={applyCNOT}>
              <Text style={styles.gateButtonText}>CNOT</Text>
              <Text style={styles.gateLabel}>Controlled-NOT</Text>
            </TouchableOpacity>
          </View>

          <Text style={styles.sectionTitle}>Pre-Built Circuits</Text>
          <View style={styles.buttonRow}>
            <TouchableOpacity style={[styles.gateButton, styles.specialButton]} onPress={createBell}>
              <Text style={styles.gateButtonText}>Bell</Text>
              <Text style={styles.gateLabel}>Entangled State</Text>
            </TouchableOpacity>
          </View>

          <Text style={styles.sectionTitle}>Operations</Text>
          <View style={styles.buttonRow}>
            <TouchableOpacity style={[styles.actionButton, styles.measureButton]} onPress={measure}>
              <Text style={styles.actionButtonText}>📏 Measure</Text>
            </TouchableOpacity>
            <TouchableOpacity style={[styles.actionButton, styles.resetButton]} onPress={reset}>
              <Text style={styles.actionButtonText}>🔄 Reset</Text>
            </TouchableOpacity>
          </View>
        </View>

        <View style={styles.output}>
          <Text style={styles.outputTitle}>Simulation Output</Text>
          <ScrollView style={styles.outputScroll} nestedScrollEnabled>
            <Text style={styles.outputText}>{output}</Text>
          </ScrollView>
        </View>
      </ScrollView>
    </LinearGradient>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
  },
  scrollView: {
    flex: 1,
  },
  scrollContent: {
    padding: 20,
    paddingBottom: 40,
  },
  header: {
    alignItems: 'center',
    marginBottom: 20,
  },
  headerTitle: {
    fontSize: 28,
    fontWeight: 'bold',
    color: '#ffffff',
    marginBottom: 8,
  },
  headerSubtitle: {
    fontSize: 14,
    color: 'rgba(255, 255, 255, 0.8)',
  },
  controls: {
    marginBottom: 20,
  },
  sectionTitle: {
    fontSize: 16,
    fontWeight: 'bold',
    color: '#ffffff',
    marginTop: 15,
    marginBottom: 10,
  },
  buttonRow: {
    flexDirection: 'row',
    gap: 10,
    marginBottom: 10,
  },
  gateButton: {
    flex: 1,
    backgroundColor: 'rgba(255, 255, 255, 0.2)',
    padding: 15,
    borderRadius: 12,
    alignItems: 'center',
  },
  gateButtonText: {
    fontSize: 24,
    fontWeight: 'bold',
    color: '#ffffff',
    marginBottom: 5,
  },
  gateLabel: {
    fontSize: 11,
    color: 'rgba(255, 255, 255, 0.8)',
  },
  specialButton: {
    backgroundColor: 'rgba(255, 215, 0, 0.3)',
  },
  actionButton: {
    flex: 1,
    padding: 15,
    borderRadius: 12,
    alignItems: 'center',
  },
  measureButton: {
    backgroundColor: 'rgba(100, 200, 100, 0.5)',
  },
  resetButton: {
    backgroundColor: 'rgba(255, 100, 100, 0.5)',
  },
  actionButtonText: {
    fontSize: 16,
    fontWeight: 'bold',
    color: '#ffffff',
  },
  output: {
    backgroundColor: 'rgba(0, 0, 0, 0.3)',
    borderRadius: 12,
    padding: 15,
    height: 300,
  },
  outputTitle: {
    fontSize: 16,
    fontWeight: 'bold',
    color: '#ffffff',
    marginBottom: 10,
  },
  outputScroll: {
    flex: 1,
  },
  outputText: {
    fontSize: 12,
    color: '#ffffff',
    fontFamily: 'monospace',
  },
});
