/**
 * RFT Visualizer Screen
 * Visualization of Resonance Fourier Transform
 */

import React, { useState } from 'react';
import {
  View,
  Text,
  StyleSheet,
  TouchableOpacity,
  ScrollView,
  ActivityIndicator,
} from 'react-native';
import ScreenShell from '../components/ScreenShell';
import { CanonicalTrueRFT, validateRFTProperties } from '../algorithms/rft/RFTCore';
import { Complex } from '../algorithms/rft/Complex';
import {
  borderRadius,
  colors,
  shadows,
  spacing,
  typography,
} from '../constants/DesignSystem';

export default function RFTVisualizerScreen() {
  const [loading, setLoading] = useState(false);
  const [output, setOutput] = useState<string>('RFT Visualizer ready. Select a transform size to begin.');
  const [validationResults, setValidationResults] = useState<any>(null);

  const appendOutput = (text: string) => {
    setOutput(prev => prev + '\n\n' + text);
  };

  const runRFT = async (size: number) => {
    setLoading(true);
    try {
      appendOutput(`=== RFT Transform (Size: ${size}) ===`);
      appendOutput('Initializing RFT with golden-ratio parameterization...');

      const rft = new CanonicalTrueRFT(size);
      appendOutput(`✓ RFT matrix constructed (${size}x${size})`);

      // Test signal
      const testSignal: Complex[] = Array(size)
        .fill(0)
        .map((_, i) => ({
          re: Math.sin(2 * Math.PI * i / size),
          im: Math.cos(2 * Math.PI * i / size),
        }));

      appendOutput('Test signal: Pure frequency wave');

      // Forward transform
      const transformed = rft.forwardTransform(testSignal);
      appendOutput(`✓ Forward transform complete`);

      // Inverse transform
      const reconstructed = rft.inverseTransform(transformed);
      appendOutput(`✓ Inverse transform complete`);

      // Calculate error
      let error = 0;
      for (let i = 0; i < size; i++) {
        const diff = Math.sqrt(
          (testSignal[i].re - reconstructed[i].re) ** 2 +
          (testSignal[i].im - reconstructed[i].im) ** 2
        );
        error += diff ** 2;
      }
      error = Math.sqrt(error);

      appendOutput(`Reconstruction error: ${error.toExponential(2)}`);
      appendOutput(`Unitarity error: ${rft.getUnitarityError().toExponential(2)}`);

      if (error < 1e-10) {
        appendOutput('✓ Transform is reversible with excellent precision');
      }
    } catch (error) {
      appendOutput(`Error: ${error}`);
    } finally {
      setLoading(false);
    }
  };

  const runValidation = async () => {
    setLoading(true);
    try {
      appendOutput('=== RFT Validation Tests ===');
      appendOutput('Running comprehensive validation...');

      const results = validateRFTProperties(32);
      setValidationResults(results);

      appendOutput(`\nValidation Results (Size: ${results.size}):`);
      appendOutput(`• Unitarity error: ${results.unitarityError.toExponential(2)}`);
      appendOutput(`• Max roundtrip error: ${results.maxRoundtripError.toExponential(2)}`);
      appendOutput(`• DFT distance: ${results.dftDistance.toFixed(3)}`);

      appendOutput('\nPaper Validation:');
      appendOutput(`• Unitarity meets spec (<1e-12): ${results.paperValidation.unitarityMeetsSpec ? '✓' : '✗'}`);
      appendOutput(`• Roundtrip acceptable (<1e-10): ${results.paperValidation.roundtripAcceptable ? '✓' : '✗'}`);
      appendOutput(`• Distinct from DFT (>1.0): ${results.paperValidation.mathematicallyDistinctFromDft ? '✓' : '✗'}`);

      if (
        results.paperValidation.unitarityMeetsSpec &&
        results.paperValidation.roundtripAcceptable &&
        results.paperValidation.mathematicallyDistinctFromDft
      ) {
        appendOutput('\n✓ ALL VALIDATIONS PASSED');
      }
    } catch (error) {
      appendOutput(`Error: ${error}`);
    } finally {
      setLoading(false);
    }
  };

  return (
    <ScreenShell
      title="RFT Visualizer"
      subtitle="Golden ratio resonance transform exploration"
    >
      <View style={styles.leadCopy}>
        <Text style={styles.leadText}>
          Execute Φ-RFT transforms at multiple orders and review validation metrics. This
          view mirrors the QuantoniumOS desktop diagnostics panel.
        </Text>
      </View>

      <View style={styles.sectionCard}>
        <Text style={styles.sectionTitle}>Transform Sizes</Text>
        <Text style={styles.sectionSubtitle}>Select a dimension to run forward and inverse passes.</Text>
        <View style={styles.buttonGrid}>
          {[8, 16, 32, 64, 128].map(size => (
            <TouchableOpacity
              key={size}
              style={[styles.sizeButton, loading && styles.buttonDisabled]}
              onPress={() => runRFT(size)}
              disabled={loading}
            >
              <Text style={styles.sizeButtonLabel}>{`${size}×${size}`}</Text>
            </TouchableOpacity>
          ))}
        </View>
      </View>

      <View style={styles.sectionCard}>
        <Text style={styles.sectionTitle}>Validation Suite</Text>
        <Text style={styles.sectionSubtitle}>
          Run the Φ-RFT paper validation harness for unitarity, roundtrip error, and DFT
          distance checks.
        </Text>
        <TouchableOpacity
          style={[styles.primaryButton, loading && styles.buttonDisabled]}
          onPress={runValidation}
          disabled={loading}
        >
          <Text style={styles.primaryButtonText}>
            {loading ? 'Running…' : 'Run Comprehensive Validation'}
          </Text>
        </TouchableOpacity>
      </View>

      {loading ? (
        <View style={styles.loadingCard}>
          <ActivityIndicator size="small" color={colors.primary} />
          <Text style={styles.loadingText}>Computing transforms…</Text>
        </View>
      ) : null}

      <View style={styles.logCard}>
        <Text style={styles.sectionTitle}>Results</Text>
        <Text style={styles.sectionSubtitle}>Logs mirror the desktop diagnostic console.</Text>
        <ScrollView style={styles.outputScroll} nestedScrollEnabled>
          <Text style={styles.outputText}>{output}</Text>
        </ScrollView>
      </View>

      <View style={styles.infoCard}>
        <Text style={styles.sectionTitle}>About Φ-RFT</Text>
        <Text style={styles.infoBody}>
          The Resonance Fourier Transform employs φ-symmetric scaling to preserve
          unitarity (&lt;1e-12 error) while remaining distinct from the Discrete Fourier
          Transform. All mobile calculations follow the published desktop reference.
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
  },
  sectionSubtitle: {
    marginTop: spacing.xs,
    marginBottom: spacing.lg,
    fontSize: typography.small,
    color: colors.gray,
  },
  buttonGrid: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    justifyContent: 'space-between',
    rowGap: spacing.sm,
    columnGap: spacing.sm,
  },
  sizeButton: {
    flexBasis: '48%',
    backgroundColor: colors.white,
    paddingVertical: spacing.md,
    borderRadius: borderRadius.lg,
    alignItems: 'center',
    borderWidth: 1,
    borderColor: 'rgba(52, 152, 219, 0.22)',
    ...shadows.sm,
  },
  buttonDisabled: {
    opacity: 0.4,
  },
  sizeButtonLabel: {
    fontSize: typography.body,
    color: colors.dark,
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
  loadingCard: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    padding: spacing.md,
    borderRadius: borderRadius.md,
    backgroundColor: colors.offWhite,
    borderWidth: 1,
    borderColor: 'rgba(52, 152, 219, 0.2)',
    marginBottom: spacing.xl,
    ...shadows.sm,
  },
  loadingText: {
    marginLeft: spacing.md,
    fontSize: typography.small,
    color: colors.darkGray,
  },
  logCard: {
    backgroundColor: colors.surfaceElevated,
    borderRadius: borderRadius.xl,
    padding: spacing.xl,
    borderWidth: 1,
    borderColor: 'rgba(52, 152, 219, 0.18)',
    ...shadows.sm,
  },
  outputScroll: {
    marginTop: spacing.sm,
    maxHeight: 280,
  },
  outputText: {
    fontSize: typography.micro,
    color: colors.dark,
    fontFamily: 'monospace',
    lineHeight: typography.micro + 6,
  },
  infoCard: {
    marginTop: spacing.xl,
    backgroundColor: colors.surfaceElevated,
    borderRadius: borderRadius.xl,
    padding: spacing.xl,
    borderWidth: 1,
    borderColor: 'rgba(52, 152, 219, 0.12)',
    ...shadows.sm,
  },
  infoBody: {
    fontSize: typography.small,
    color: colors.darkGray,
    lineHeight: typography.small + 6,
  },
});
