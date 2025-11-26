/**
 * Validation Screen - System Status and Testing
 */

import React, { useState, useEffect } from 'react';
import {
  View,
  Text,
  StyleSheet,
  TouchableOpacity,
  ScrollView,
  ActivityIndicator,
} from 'react-native';
import { LinearGradient } from 'expo-linear-gradient';
import { CanonicalTrueRFT } from '../algorithms/rft/RFTCore';
import { RFTEnhancedFeistel } from '../algorithms/crypto/CryptoPrimitives';
import { QuantumCircuits } from '../algorithms/quantum/QuantumSimulator';
import { measureCHSH } from '../algorithms/quantum/QuantumGates';

interface TestResult {
  name: string;
  status: 'pending' | 'running' | 'passed' | 'failed';
  message: string;
  time?: number;
}

export default function ValidationScreen() {
  const [tests, setTests] = useState<TestResult[]>([
    { name: 'RFT Unitarity', status: 'pending', message: 'Not run' },
    { name: 'RFT Round-trip', status: 'pending', message: 'Not run' },
    { name: 'Crypto Encryption', status: 'pending', message: 'Not run' },
    { name: 'Crypto Decryption', status: 'pending', message: 'Not run' },
    { name: 'Quantum Bell State', status: 'pending', message: 'Not run' },
    { name: 'Quantum CHSH Test', status: 'pending', message: 'Not run' },
  ]);
  const [running, setRunning] = useState(false);
  const [overallStatus, setOverallStatus] = useState<'idle' | 'testing' | 'passed' | 'failed'>('idle');

  const updateTest = (index: number, update: Partial<TestResult>) => {
    setTests(prev => prev.map((t, i) => (i === index ? { ...t, ...update } : t)));
  };

  const runAllTests = async () => {
    setRunning(true);
    setOverallStatus('testing');

    try {
      // Test 1: RFT Unitarity
      updateTest(0, { status: 'running', message: 'Testing...' });
      const startTime1 = Date.now();
      const rft = new CanonicalTrueRFT(32);
      const unitarityError = rft.getUnitarityError();
      const time1 = Date.now() - startTime1;

      if (unitarityError < 1e-12) {
        updateTest(0, {
          status: 'passed',
          message: `Error: ${unitarityError.toExponential(2)} (< 1e-12)`,
          time: time1,
        });
      } else {
        updateTest(0, {
          status: 'failed',
          message: `Error: ${unitarityError.toExponential(2)} (too high)`,
          time: time1,
        });
      }

      // Test 2: RFT Round-trip
      updateTest(1, { status: 'running', message: 'Testing...' });
      const startTime2 = Date.now();
      const testSignal = Array(32).fill(0).map(() => ({
        re: Math.random(),
        im: Math.random(),
      }));
      const transformed = rft.forwardTransform(testSignal);
      const reconstructed = rft.inverseTransform(transformed);

      let roundtripError = 0;
      for (let i = 0; i < 32; i++) {
        const diff = Math.sqrt(
          (testSignal[i].re - reconstructed[i].re) ** 2 +
          (testSignal[i].im - reconstructed[i].im) ** 2
        );
        roundtripError += diff ** 2;
      }
      roundtripError = Math.sqrt(roundtripError);
      const time2 = Date.now() - startTime2;

      if (roundtripError < 1e-10) {
        updateTest(1, {
          status: 'passed',
          message: `Error: ${roundtripError.toExponential(2)} (< 1e-10)`,
          time: time2,
        });
      } else {
        updateTest(1, {
          status: 'failed',
          message: `Error: ${roundtripError.toExponential(2)} (too high)`,
          time: time2,
        });
      }

      // Test 3: Crypto Encryption
      updateTest(2, { status: 'running', message: 'Testing...' });
      const startTime3 = Date.now();
      const cipher = new RFTEnhancedFeistel(48, 16);
      const key = new Uint8Array(32).fill(42);
      const plaintext = new Uint8Array(16).fill(0x5a);
      const ciphertext = cipher.encrypt(plaintext, key);
      const time3 = Date.now() - startTime3;

      if (ciphertext.length === 16 && !ciphertext.every((b, i) => b === plaintext[i])) {
        updateTest(2, {
          status: 'passed',
          message: `Encrypted 16 bytes (${ciphertext.slice(0, 4).join(',')})`,
          time: time3,
        });
      } else {
        updateTest(2, {
          status: 'failed',
          message: 'Encryption did not change data',
          time: time3,
        });
      }

      // Test 4: Crypto Decryption
      updateTest(3, { status: 'running', message: 'Testing...' });
      const startTime4 = Date.now();
      const decrypted = cipher.decrypt(ciphertext, key);
      const time4 = Date.now() - startTime4;

      const decryptionValid = plaintext.every((b, i) => b === decrypted[i]);
      if (decryptionValid) {
        updateTest(3, {
          status: 'passed',
          message: 'Successfully decrypted to original',
          time: time4,
        });
      } else {
        updateTest(3, {
          status: 'failed',
          message: 'Decryption mismatch',
          time: time4,
        });
      }

      // Test 5: Quantum Bell State
      updateTest(4, { status: 'running', message: 'Testing...' });
      const startTime5 = Date.now();
      const bellSim = QuantumCircuits.bellState();
      const probs = bellSim.getProbabilities();
      const time5 = Date.now() - startTime5;

      const hasCorrectStates = probs.some(p => p.state === '|00⟩') && probs.some(p => p.state === '|11⟩');
      const probsCorrect = probs.every(p => Math.abs(p.probability - 0.5) < 0.01);

      if (hasCorrectStates && probsCorrect) {
        updateTest(4, {
          status: 'passed',
          message: '|00⟩ and |11⟩ with ~50% each',
          time: time5,
        });
      } else {
        updateTest(4, {
          status: 'failed',
          message: 'Incorrect Bell state probabilities',
          time: time5,
        });
      }

      // Test 6: Quantum CHSH
      updateTest(5, { status: 'running', message: 'Testing...' });
      const startTime6 = Date.now();
      const chsh = measureCHSH();
      const time6 = Date.now() - startTime6;

      if (Math.abs(chsh - 2.828) < 0.01) {
        updateTest(5, {
          status: 'passed',
          message: `CHSH = ${chsh.toFixed(3)} (violates classical limit)`,
          time: time6,
        });
      } else {
        updateTest(5, {
          status: 'failed',
          message: `CHSH = ${chsh.toFixed(3)} (expected 2.828)`,
          time: time6,
        });
      }

      // Determine overall status
      setTests(currentTests => {
        const allPassed = currentTests.every(t => t.status === 'passed');
        setOverallStatus(allPassed ? 'passed' : 'failed');
        return currentTests;
      });
    } catch (error) {
      console.error('Test error:', error);
      setOverallStatus('failed');
    } finally {
      setRunning(false);
    }
  };

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'passed':
        return '#4caf50';
      case 'failed':
        return '#f44336';
      case 'running':
        return '#ff9800';
      default:
        return '#9e9e9e';
    }
  };

  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'passed':
        return '✓';
      case 'failed':
        return '✗';
      case 'running':
        return '⟳';
      default:
        return '○';
    }
  };

  return (
    <LinearGradient colors={['#fa709a', '#fee140']} style={styles.container}>
      <ScrollView style={styles.scrollView} contentContainerStyle={styles.scrollContent}>
        <View style={styles.header}>
          <Text style={styles.headerTitle}>System Validation</Text>
          <Text style={styles.headerSubtitle}>Comprehensive Testing Suite</Text>
        </View>

        <View style={styles.overallStatus}>
          <Text style={styles.overallStatusTitle}>Overall Status</Text>
          <View
            style={[
              styles.overallStatusBadge,
              { backgroundColor: getStatusColor(overallStatus) },
            ]}
          >
            <Text style={styles.overallStatusText}>
              {overallStatus === 'idle' && 'Ready to Test'}
              {overallStatus === 'testing' && 'Testing in Progress...'}
              {overallStatus === 'passed' && 'ALL TESTS PASSED'}
              {overallStatus === 'failed' && 'SOME TESTS FAILED'}
            </Text>
          </View>
        </View>

        <TouchableOpacity
          style={[styles.runButton, running && styles.runButtonDisabled]}
          onPress={runAllTests}
          disabled={running}
        >
          {running ? (
            <ActivityIndicator color="#ffffff" />
          ) : (
            <Text style={styles.runButtonText}>▶ Run All Tests</Text>
          )}
        </TouchableOpacity>

        <View style={styles.testsList}>
          {tests.map((test, index) => (
            <View
              key={index}
              style={[
                styles.testCard,
                { borderLeftColor: getStatusColor(test.status), borderLeftWidth: 4 },
              ]}
            >
              <View style={styles.testHeader}>
                <Text style={styles.testIcon}>{getStatusIcon(test.status)}</Text>
                <View style={styles.testInfo}>
                  <Text style={styles.testName}>{test.name}</Text>
                  <Text style={styles.testMessage}>{test.message}</Text>
                  {test.time && (
                    <Text style={styles.testTime}>{test.time}ms</Text>
                  )}
                </View>
              </View>
            </View>
          ))}
        </View>

        <View style={styles.footer}>
          <Text style={styles.footerText}>
            QuantoniumOS Mobile v1.0
          </Text>
          <Text style={styles.footerSubtext}>
            Patent-Pending USPTO 19/169,399
          </Text>
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
  overallStatus: {
    backgroundColor: 'rgba(255, 255, 255, 0.2)',
    padding: 20,
    borderRadius: 12,
    alignItems: 'center',
    marginBottom: 20,
  },
  overallStatusTitle: {
    fontSize: 16,
    fontWeight: 'bold',
    color: '#ffffff',
    marginBottom: 10,
  },
  overallStatusBadge: {
    paddingHorizontal: 20,
    paddingVertical: 10,
    borderRadius: 20,
  },
  overallStatusText: {
    fontSize: 16,
    fontWeight: 'bold',
    color: '#ffffff',
  },
  runButton: {
    backgroundColor: '#2196f3',
    padding: 15,
    borderRadius: 12,
    alignItems: 'center',
    marginBottom: 20,
  },
  runButtonDisabled: {
    backgroundColor: '#9e9e9e',
  },
  runButtonText: {
    fontSize: 16,
    fontWeight: 'bold',
    color: '#ffffff',
  },
  testsList: {
    gap: 12,
    marginBottom: 20,
  },
  testCard: {
    backgroundColor: 'rgba(255, 255, 255, 0.9)',
    padding: 15,
    borderRadius: 8,
  },
  testHeader: {
    flexDirection: 'row',
    alignItems: 'flex-start',
  },
  testIcon: {
    fontSize: 24,
    marginRight: 12,
  },
  testInfo: {
    flex: 1,
  },
  testName: {
    fontSize: 16,
    fontWeight: 'bold',
    color: '#333',
    marginBottom: 4,
  },
  testMessage: {
    fontSize: 13,
    color: '#666',
  },
  testTime: {
    fontSize: 11,
    color: '#999',
    marginTop: 4,
  },
  footer: {
    alignItems: 'center',
    marginTop: 20,
  },
  footerText: {
    fontSize: 14,
    color: 'rgba(255, 255, 255, 0.9)',
    fontWeight: '600',
  },
  footerSubtext: {
    fontSize: 12,
    color: 'rgba(255, 255, 255, 0.7)',
    marginTop: 5,
  },
});
