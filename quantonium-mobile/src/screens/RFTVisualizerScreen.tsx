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
import { LinearGradient } from 'expo-linear-gradient';
import { CanonicalTrueRFT, validateRFTProperties } from '../algorithms/rft/RFTCore';
import { Complex } from '../algorithms/rft/Complex';

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
    <LinearGradient colors={['#43e97b', '#38f9d7']} style={styles.container}>
      <ScrollView style={styles.scrollView} contentContainerStyle={styles.scrollContent}>
        <View style={styles.header}>
          <Text style={styles.headerTitle}>📊 RFT Visualizer</Text>
          <Text style={styles.headerSubtitle}>Golden-Ratio Parameterized Transform</Text>
        </View>

        <View style={styles.controls}>
          <Text style={styles.sectionTitle}>Transform Sizes</Text>
          <View style={styles.buttonRow}>
            <TouchableOpacity
              style={styles.sizeButton}
              onPress={() => runRFT(8)}
              disabled={loading}
            >
              <Text style={styles.sizeButtonText}>8×8</Text>
            </TouchableOpacity>
            <TouchableOpacity
              style={styles.sizeButton}
              onPress={() => runRFT(16)}
              disabled={loading}
            >
              <Text style={styles.sizeButtonText}>16×16</Text>
            </TouchableOpacity>
            <TouchableOpacity
              style={styles.sizeButton}
              onPress={() => runRFT(32)}
              disabled={loading}
            >
              <Text style={styles.sizeButtonText}>32×32</Text>
            </TouchableOpacity>
          </View>

          <View style={styles.buttonRow}>
            <TouchableOpacity
              style={styles.sizeButton}
              onPress={() => runRFT(64)}
              disabled={loading}
            >
              <Text style={styles.sizeButtonText}>64×64</Text>
            </TouchableOpacity>
            <TouchableOpacity
              style={styles.sizeButton}
              onPress={() => runRFT(128)}
              disabled={loading}
            >
              <Text style={styles.sizeButtonText}>128×128</Text>
            </TouchableOpacity>
          </View>

          <Text style={styles.sectionTitle}>Validation</Text>
          <TouchableOpacity
            style={[styles.actionButton, styles.validateButton]}
            onPress={runValidation}
            disabled={loading}
          >
            <Text style={styles.actionButtonText}>
              {loading ? 'Running...' : '✓ Run Full Validation'}
            </Text>
          </TouchableOpacity>
        </View>

        {loading && (
          <View style={styles.loadingContainer}>
            <ActivityIndicator size="large" color="#ffffff" />
            <Text style={styles.loadingText}>Computing transforms...</Text>
          </View>
        )}

        <View style={styles.output}>
          <Text style={styles.outputTitle}>Results</Text>
          <ScrollView style={styles.outputScroll} nestedScrollEnabled>
            <Text style={styles.outputText}>{output}</Text>
          </ScrollView>
        </View>

        <View style={styles.infoBox}>
          <Text style={styles.infoTitle}>About RFT</Text>
          <Text style={styles.infoText}>
            The Resonance Fourier Transform uses golden ratio (φ = 1.618...)
            parameterization in its kernel matrix. It achieves unitarity error
            {'<'}1e-12 and is mathematically distinct from the DFT.
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
    textAlign: 'center',
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
  sizeButton: {
    flex: 1,
    backgroundColor: 'rgba(255, 255, 255, 0.3)',
    padding: 15,
    borderRadius: 12,
    alignItems: 'center',
  },
  sizeButtonText: {
    fontSize: 16,
    fontWeight: 'bold',
    color: '#ffffff',
  },
  actionButton: {
    padding: 15,
    borderRadius: 12,
    alignItems: 'center',
  },
  validateButton: {
    backgroundColor: 'rgba(100, 150, 255, 0.5)',
  },
  actionButtonText: {
    fontSize: 16,
    fontWeight: 'bold',
    color: '#ffffff',
  },
  loadingContainer: {
    alignItems: 'center',
    padding: 20,
  },
  loadingText: {
    color: '#ffffff',
    marginTop: 10,
    fontSize: 14,
  },
  output: {
    backgroundColor: 'rgba(0, 0, 0, 0.3)',
    borderRadius: 12,
    padding: 15,
    height: 300,
    marginBottom: 20,
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
    fontSize: 11,
    color: '#ffffff',
    fontFamily: 'monospace',
  },
  infoBox: {
    backgroundColor: 'rgba(255, 255, 255, 0.2)',
    padding: 15,
    borderRadius: 12,
  },
  infoTitle: {
    fontSize: 16,
    fontWeight: 'bold',
    color: '#ffffff',
    marginBottom: 8,
  },
  infoText: {
    fontSize: 13,
    color: 'rgba(255, 255, 255, 0.9)',
    lineHeight: 20,
  },
});
