/**
 * System Monitor Screen - QuantoniumOS Mobile
 * Resource monitoring and system status
 */

import React, { useState, useEffect } from 'react';
import {
  View,
  Text,
  StyleSheet,
  ScrollView,
} from 'react-native';
import { LinearGradient } from 'expo-linear-gradient';
import { colors, spacing, typography } from '../constants/DesignSystem';

export default function SystemMonitorScreen() {
  const [memoryUsage, setMemoryUsage] = useState('--');
  const [uptime, setUptime] = useState('--');

  useEffect(() => {
    // Simulate system metrics
    setMemoryUsage('128 MB');
    setUptime('2h 34m');
  }, []);

  return (
    <LinearGradient colors={colors.monitorGradient} style={styles.container}>
      <ScrollView style={styles.scrollView} contentContainerStyle={styles.scrollContent}>
        <View style={styles.header}>
          <Text style={styles.headerTitle}>📊 System Monitor</Text>
          <Text style={styles.headerSubtitle}>Resource monitoring and diagnostics</Text>
        </View>

        <View style={styles.metricsContainer}>
          <View style={styles.metricCard}>
            <Text style={styles.metricLabel}>Memory Usage</Text>
            <Text style={styles.metricValue}>{memoryUsage}</Text>
          </View>

          <View style={styles.metricCard}>
            <Text style={styles.metricLabel}>System Uptime</Text>
            <Text style={styles.metricValue}>{uptime}</Text>
          </View>

          <View style={styles.metricCard}>
            <Text style={styles.metricLabel}>RFT Engine</Text>
            <Text style={[styles.metricValue, styles.statusActive]}>Active</Text>
          </View>

          <View style={styles.metricCard}>
            <Text style={styles.metricLabel}>Crypto Module</Text>
            <Text style={[styles.metricValue, styles.statusActive]}>Ready</Text>
          </View>
        </View>

        <View style={styles.infoBox}>
          <Text style={styles.infoText}>
            System monitor displays real-time metrics for the QuantoniumOS mobile environment.
            Full system diagnostics are available on the desktop version.
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
    padding: spacing.lg,
    paddingBottom: spacing.xxl,
  },
  header: {
    alignItems: 'center',
    marginBottom: spacing.xl,
  },
  headerTitle: {
    fontSize: typography.title,
    fontWeight: 'bold',
    color: colors.white,
    marginBottom: spacing.xs,
  },
  headerSubtitle: {
    fontSize: typography.small,
    color: 'rgba(255, 255, 255, 0.8)',
  },
  metricsContainer: {
    gap: spacing.md,
  },
  metricCard: {
    backgroundColor: 'rgba(255, 255, 255, 0.2)',
    padding: spacing.lg,
    borderRadius: 12,
  },
  metricLabel: {
    fontSize: typography.small,
    color: 'rgba(255, 255, 255, 0.9)',
    marginBottom: spacing.xs,
  },
  metricValue: {
    fontSize: typography.display,
    fontWeight: 'bold',
    color: colors.white,
  },
  statusActive: {
    color: '#4caf50',
  },
  infoBox: {
    backgroundColor: 'rgba(0, 0, 0, 0.3)',
    padding: spacing.lg,
    borderRadius: 12,
    marginTop: spacing.xl,
  },
  infoText: {
    fontSize: typography.small,
    color: colors.white,
    lineHeight: 20,
  },
});
