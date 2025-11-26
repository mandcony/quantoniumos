/**
 * Validation Screen - QuantoniumOS Mobile
 * Mirrors the desktop Φ-RFT validation harness
 */

import React, { useState } from 'react';
import {
  ActivityIndicator,
  StyleSheet,
  Text,
  TouchableOpacity,
  View,
} from 'react-native';
import ScreenShell from '../components/ScreenShell';
import { CanonicalTrueRFT } from '../algorithms/rft/RFTCore';
import { RFTEnhancedFeistel } from '../algorithms/crypto/CryptoPrimitives';
import { QuantumCircuits } from '../algorithms/quantum/QuantumSimulator';
import { createBellState, measureCHSH } from '../algorithms/quantum/QuantumGates';
import { VARIANTS, validateVariantMatrix } from '../algorithms/rft/VariantRegistry';
import { subtract, abs } from '../algorithms/rft/Complex';
import {
  borderRadius,
  colors,
  shadows,
  spacing,
  typography,
} from '../constants/DesignSystem';

interface TestResult {
  name: string;
  status: 'pending' | 'running' | 'passed' | 'failed';
  message: string;
  time?: number;
}

type OverallStatus = 'idle' | 'testing' | 'passed' | 'failed';

type StatusToken = {
  background: string;
  foreground: string;
  label: string;
};

const STATUS_PALETTE: Record<TestResult['status'] | OverallStatus, StatusToken> = {
  pending: {
    background: 'rgba(52, 152, 219, 0.08)',
    foreground: colors.darkGray,
    label: 'Pending',
  },
  running: {
    background: 'rgba(243, 156, 18, 0.18)',
    foreground: colors.warning,
    label: 'Running',
  },
  passed: {
    background: 'rgba(39, 174, 96, 0.16)',
    foreground: colors.success,
    label: 'Passed',
  },
  failed: {
    background: 'rgba(231, 76, 60, 0.16)',
    foreground: colors.error,
    label: 'Failed',
  },
  idle: {
    background: 'rgba(52, 152, 219, 0.12)',
    foreground: colors.primary,
    label: 'Idle',
  },
  testing: {
    background: 'rgba(243, 156, 18, 0.14)',
    foreground: colors.warning,
    label: 'Testing',
  },
};

const INITIAL_TESTS: TestResult[] = [
  { name: 'RFT Unitarity', status: 'pending', message: 'Not run' },
  { name: 'RFT Round-trip', status: 'pending', message: 'Not run' },
  { name: 'Crypto Encryption', status: 'pending', message: 'Not run' },
  { name: 'Crypto Decryption', status: 'pending', message: 'Not run' },
  { name: 'Quantum Bell State', status: 'pending', message: 'Not run' },
  { name: 'Quantum CHSH', status: 'pending', message: 'Not run' },
  { name: 'Φ-RFT Variant Family', status: 'pending', message: 'Not run' },
];

function getStatusToken(status: TestResult['status'] | OverallStatus) {
  return STATUS_PALETTE[status];
}

function formatDuration(ms?: number) {
  if (typeof ms !== 'number') {
    return '';
  }
  if (ms < 1000) {
    return `${ms} ms`;
  }
  const seconds = ms / 1000;
  return `${seconds.toFixed(2)} s`;
}

export default function ValidationScreen() {
  const [tests, setTests] = useState<TestResult[]>(INITIAL_TESTS);
  const [overallStatus, setOverallStatus] = useState<OverallStatus>('idle');
  const [running, setRunning] = useState(false);

  const runAllTests = async () => {
    if (running) {
      return;
    }

    setRunning(true);
    setOverallStatus('testing');

    const nextTests = INITIAL_TESTS.map(test => ({ ...test }));
    setTests(nextTests);

    try {
      const rft = new CanonicalTrueRFT(32);
      const cipher = new RFTEnhancedFeistel(48, 16);

      // Test 0: RFT Unitarity
      nextTests[0] = { ...nextTests[0], status: 'running', message: 'Computing Φ residual' };
      setTests([...nextTests]);
      let started = Date.now();
      const unitarityError = rft.getUnitarityError();
      let elapsed = Date.now() - started;
      const unitarityPass = unitarityError < 1e-12;
      nextTests[0] = {
        ...nextTests[0],
        status: unitarityPass ? 'passed' : 'failed',
        message: `ε = ${unitarityError.toExponential(2)}`,
        time: elapsed,
      };
      setTests([...nextTests]);

      // Test 1: RFT Round-trip
      nextTests[1] = { ...nextTests[1], status: 'running', message: 'Verifying forward/inverse fidelity' };
      setTests([...nextTests]);
      started = Date.now();
      const signal = Array.from({ length: 32 }, (_, index) => ({
        re: Math.sin(index * 0.91),
        im: Math.cos(index * 1.13),
      }));
      const transformed = rft.forwardTransform(signal);
      const reconstructed = rft.inverseTransform(transformed);
      let maxError = 0;
      for (let i = 0; i < signal.length; i += 1) {
        const error = abs(subtract(signal[i], reconstructed[i]));
        if (error > maxError) {
          maxError = error;
        }
      }
      elapsed = Date.now() - started;
      const roundtripPass = maxError < 1e-10;
      nextTests[1] = {
        ...nextTests[1],
        status: roundtripPass ? 'passed' : 'failed',
        message: `δ = ${maxError.toExponential(2)}`,
        time: elapsed,
      };
      setTests([...nextTests]);

      // Prepare deterministic block/key for crypto tests
      const key = new Uint8Array(32);
      for (let i = 0; i < key.length; i += 1) {
        key[i] = (i * 37) % 256;
      }
      const plaintext = new Uint8Array(16);
      for (let i = 0; i < plaintext.length; i += 1) {
        plaintext[i] = (i * 11) % 256;
      }

      // Test 2: Crypto Encryption
      nextTests[2] = { ...nextTests[2], status: 'running', message: 'Executing 48-round Feistel cipher' };
      setTests([...nextTests]);
      started = Date.now();
      const ciphertext = cipher.encrypt(plaintext, key);
      elapsed = Date.now() - started;
      const encryptionDiffers = ciphertext.some((byte, idx) => byte !== plaintext[idx]);
      nextTests[2] = {
        ...nextTests[2],
        status: encryptionDiffers ? 'passed' : 'failed',
        message: encryptionDiffers
          ? `Ciphertext diverges (0x${ciphertext[0].toString(16).padStart(2, '0')})`
          : 'Ciphertext matches plaintext – check cipher',
        time: elapsed,
      };
      setTests([...nextTests]);

      // Test 3: Crypto Decryption
      nextTests[3] = { ...nextTests[3], status: 'running', message: 'Checking decryption parity' };
      setTests([...nextTests]);
      started = Date.now();
      const decrypted = cipher.decrypt(ciphertext, key);
      elapsed = Date.now() - started;
      const cryptoParity = decrypted.every((byte, idx) => byte === plaintext[idx]);
      nextTests[3] = {
        ...nextTests[3],
        status: cryptoParity ? 'passed' : 'failed',
        message: cryptoParity ? 'Parity restored' : 'Mismatch on decrypt',
        time: elapsed,
      };
      setTests([...nextTests]);

      // Test 4: Quantum Bell State
      nextTests[4] = { ...nextTests[4], status: 'running', message: 'Preparing |Φ+⟩ and evaluating fidelity' };
      setTests([...nextTests]);
      started = Date.now();
      const bellSimulator = QuantumCircuits.bellState();
      const expectedBell = createBellState();
      const fidelity = bellSimulator.fidelity(expectedBell);
      const bellState = bellSimulator.getState();
      const prob00 = abs(bellState.amplitudes[0]) ** 2;
      elapsed = Date.now() - started;
      const bellPass = fidelity > 0.999;
      nextTests[4] = {
        ...nextTests[4],
        status: bellPass ? 'passed' : 'failed',
        message: `Fidelity = ${fidelity.toFixed(6)}; P(|00⟩) = ${prob00.toFixed(3)}`,
        time: elapsed,
      };
      setTests([...nextTests]);

      // Test 5: Quantum CHSH
      nextTests[5] = { ...nextTests[5], status: 'running', message: 'Measuring CHSH inequality bound' };
      setTests([...nextTests]);
      started = Date.now();
      const chsh = measureCHSH();
      elapsed = Date.now() - started;
      const chshPass = chsh > 2.0;
      nextTests[5] = {
        ...nextTests[5],
        status: chshPass ? 'passed' : 'failed',
        message: `S = ${chsh.toFixed(3)}`,
        time: elapsed,
      };
      setTests([...nextTests]);

      // Test 6: Φ-RFT Variant Family
      nextTests[6] = { ...nextTests[6], status: 'running', message: 'Orthonormalising Φ variants' };
      setTests([...nextTests]);
      started = Date.now();
      const subset = VARIANTS.slice(0, 5);
      const variantResiduals = subset.map(variant => {
        const matrix = variant.generator(16);
        const residual = validateVariantMatrix(matrix);
        return { key: variant.key, name: variant.name, residual };
      });
      elapsed = Date.now() - started;
      const worstResidual = Math.max(...variantResiduals.map(entry => entry.residual));
      const variantsPass = worstResidual < 1e-10;
      const summary = variantResiduals
        .map(entry => `${entry.name.split(' ')[0]}:${entry.residual.toExponential(1)}`)
        .join('  ');
      nextTests[6] = {
        ...nextTests[6],
        status: variantsPass ? 'passed' : 'failed',
        message: summary,
        time: elapsed,
      };
      setTests([...nextTests]);

      const allPassed = nextTests.every(test => test.status === 'passed');
      setOverallStatus(allPassed ? 'passed' : 'failed');
    } catch (error) {
      console.error('Validation suite failed', error);
      nextTests.forEach((test, index) => {
        if (test.status === 'running') {
          nextTests[index] = {
            ...test,
            status: 'failed',
            message: 'Execution interrupted',
          };
        }
      });
      setTests([...nextTests]);
      setOverallStatus('failed');
    } finally {
      setRunning(false);
    }
  };

  return (
    <ScreenShell
      title="RFT Validation Suite"
      subtitle="QuantoniumOS reproducibility harness"
    >
      <View style={styles.leadCopy}>
        <Text style={styles.leadText}>
          Execute the exact Φ-RFT validation regimen distributed with the desktop build.
          Every gate mirrors the reproducibility notebook to certify on-device fidelity.
        </Text>
      </View>

      <View style={styles.statusCard}>
        <Text style={styles.sectionTitle}>Overall Status</Text>
        <View
          style={[
            styles.statusPill,
            { backgroundColor: getStatusToken(overallStatus).background },
          ]}
        >
          <Text style={[styles.statusText, { color: getStatusToken(overallStatus).foreground }]}>
            {overallStatus === 'idle' && 'Ready to run the Φ-RFT harness'}
            {overallStatus === 'testing' && 'Validation in progress'}
            {overallStatus === 'passed' && 'All validation gates passed'}
            {overallStatus === 'failed' && 'Attention: at least one gate failed'}
          </Text>
        </View>

        <TouchableOpacity
          style={[styles.primaryButton, running && styles.buttonDisabled]}
          onPress={runAllTests}
          disabled={running}
        >
          {running ? (
            <ActivityIndicator color={colors.white} />
          ) : (
            <Text style={styles.primaryButtonText}>Run Full Validation</Text>
          )}
        </TouchableOpacity>
      </View>

      <View style={styles.testsList}>
        {tests.map(test => {
          const token = getStatusToken(test.status);
          return (
            <View key={test.name} style={styles.testCard}>
              <View style={styles.testHeader}>
                <View style={[styles.testIndicator, { backgroundColor: token.background }]} />
                <View style={styles.testInfo}>
                  <Text style={styles.testName}>{test.name}</Text>
                  <Text style={styles.testMessage}>{test.message}</Text>
                </View>
                <Text style={[styles.testState, { color: token.foreground }]}>
                  {token.label}
                </Text>
              </View>
              {test.time != null ? (
                <Text style={styles.testTime}> {formatDuration(test.time)}</Text>
              ) : null}
            </View>
          );
        })}
      </View>

      <View style={styles.infoCard}>
        <Text style={styles.infoTitle}>Harness Coverage</Text>
        <Text style={styles.infoText}>
          Suite checks matrix unitarity, round-trip fidelity, cipher parity, Bell-state fidelity,
          CHSH violation, and Φ-RFT variant orthonormality. Identical tolerances to the USPTO-ready
          desktop harness ensure results remain publication-grade on mobile hardware.
        </Text>
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
  statusCard: {
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
  statusPill: {
    borderRadius: borderRadius.lg,
    paddingVertical: spacing.sm,
    paddingHorizontal: spacing.lg,
    marginBottom: spacing.lg,
  },
  statusText: {
    fontSize: typography.small,
    textAlign: 'center',
    fontWeight: '600',
  },
  primaryButton: {
    backgroundColor: colors.primary,
    borderRadius: borderRadius.lg,
    paddingVertical: spacing.md,
    alignItems: 'center',
    ...shadows.md,
  },
  primaryButtonText: {
    fontSize: typography.body,
    color: colors.white,
    fontWeight: '600',
    letterSpacing: 1,
  },
  buttonDisabled: {
    opacity: 0.6,
  },
  testsList: {
    marginBottom: spacing.xl,
  },
  testCard: {
    backgroundColor: colors.surfaceElevated,
    borderRadius: borderRadius.xl,
    padding: spacing.lg,
    marginBottom: spacing.md,
    borderWidth: 1,
    borderColor: 'rgba(52, 152, 219, 0.12)',
    ...shadows.sm,
  },
  testHeader: {
    flexDirection: 'row',
    alignItems: 'flex-start',
  },
  testIndicator: {
    width: 12,
    height: 12,
    borderRadius: 6,
    marginTop: spacing.xs,
    marginRight: spacing.md,
  },
  testInfo: {
    flex: 1,
  },
  testName: {
    fontSize: typography.body,
    color: colors.dark,
    fontWeight: '600',
  },
  testMessage: {
    marginTop: spacing.xs,
    fontSize: typography.small,
    color: colors.gray,
    lineHeight: typography.small + 4,
  },
  testState: {
    fontSize: typography.small,
    fontWeight: '600',
  },
  testTime: {
    marginTop: spacing.md,
    fontSize: typography.micro,
    color: colors.gray,
    fontFamily: 'monospace',
  },
  infoCard: {
    backgroundColor: colors.surfaceElevated,
    borderRadius: borderRadius.xl,
    padding: spacing.xl,
    borderWidth: 1,
    borderColor: 'rgba(52, 152, 219, 0.12)',
    ...shadows.sm,
  },
  infoTitle: {
    fontSize: typography.subtitle,
    color: colors.dark,
    fontWeight: '600',
    marginBottom: spacing.sm,
  },
  infoText: {
    fontSize: typography.small,
    color: colors.darkGray,
    lineHeight: typography.small + 6,
  },
});
