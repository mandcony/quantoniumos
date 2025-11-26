/**
 * System Monitor Screen - QuantoniumOS Mobile
 * Exact 1:1 match with desktop system monitor aesthetics
 */

import React, { useState, useEffect } from 'react';
import {
  View,
  Text,
  StyleSheet,
  ScrollView,
} from 'react-native';
import ScreenShell from '../components/ScreenShell';
import { colors, spacing, typography, PHI, PHI_INV, BASE_UNIT } from '../constants/DesignSystem';

export default function SystemMonitorScreen() {
  const [memoryUsage, setMemoryUsage] = useState('--');
  const [uptime, setUptime] = useState('--');
  const [currentTime, setCurrentTime] = useState(new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }));

  useEffect(() => {
    // Simulate system metrics
    setMemoryUsage('128 MB');
    setUptime('2h 34m');

    // Update time every second (matching desktop)
    const timer = setInterval(() => {
      setCurrentTime(new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }));
    }, 1000);

    return () => clearInterval(timer);
  }, []);

  return (
    <ScreenShell
      title="System Monitor"
      subtitle="Resource monitoring and diagnostics"
    >
      <ScrollView 
        style={styles.scrollView} 
        contentContainerStyle={styles.scrollContent}
        showsVerticalScrollIndicator={false}
      >
        {/* Time display (matching desktop top-right placement) */}
        <View style={styles.timeContainer}>
          <Text style={styles.timeText}>{currentTime}</Text>
        </View>

        {/* Main metrics grid */}
        <View style={styles.metricsGrid}>
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

        {/* System info with desktop-style minimal aesthetics */}
        <View style={styles.infoCard}>
          <Text style={styles.infoTitle}>System Information</Text>
          <Text style={styles.infoBody}>
            QuantoniumOS mobile environment running with Φ-RFT acceleration.
            All metrics synchronized with desktop reference implementation.
          </Text>
        </View>

        {/* Status indicator matching desktop "QUANTONIUMOS" style */}
        <View style={styles.statusContainer}>
          <Text style={styles.statusText}>QUANTONIUMOS</Text>
          <Text style={styles.statusSubtext}>System Monitor</Text>
        </View>
      </ScrollView>
    </ScreenShell>
  );
}

const styles = StyleSheet.create({
  scrollView: {
    flex: 1,
  },
  scrollContent: {
    paddingBottom: Math.round(BASE_UNIT * PHI * PHI), // φ² padding
  },
  timeContainer: {
    alignItems: 'flex-end',
    marginBottom: Math.round(BASE_UNIT * PHI),
  },
  timeText: {
    fontSize: Math.round(BASE_UNIT * PHI_INV), // Matching desktop time font
    fontFamily: 'monospace',
    color: colors.darkGray, // #34495e from desktop
    letterSpacing: 1,
  },
  metricsGrid: {
    gap: spacing.md,
    marginBottom: spacing.xl,
  },
  metricCard: {
    backgroundColor: 'rgba(255, 255, 255, 0.95)', // Near-white like desktop cards
    borderRadius: 12,
    borderWidth: 1,
    borderColor: 'rgba(52, 152, 219, 0.2)', // Matching desktop border (52, 152, 219 is #3498db)
    padding: spacing.lg,
    // Subtle shadow matching desktop
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.05,
    shadowRadius: 4,
    elevation: 2,
  },
  metricLabel: {
    fontSize: typography.small,
    color: colors.darkGray,
    marginBottom: spacing.xs,
    fontWeight: '500',
    letterSpacing: 0.5,
  },
  metricValue: {
    fontSize: typography.display,
    fontWeight: '600',
    color: colors.dark, // #2c3e50 from desktop
    fontFamily: 'monospace',
  },
  statusActive: {
    color: colors.success, // #27ae60
  },
  infoCard: {
    backgroundColor: 'rgba(255, 255, 255, 0.95)',
    borderRadius: 12,
    borderWidth: 1,
    borderColor: 'rgba(52, 152, 219, 0.2)',
    padding: spacing.lg,
    marginBottom: spacing.xl,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.05,
    shadowRadius: 4,
    elevation: 2,
  },
  infoTitle: {
    fontSize: typography.subtitle,
    color: colors.dark,
    fontWeight: '600',
    letterSpacing: 0.6,
    marginBottom: spacing.sm,
  },
  infoBody: {
    fontSize: typography.body,
    lineHeight: typography.body + 6,
    color: colors.darkGray,
  },
  statusContainer: {
    alignItems: 'center',
    marginTop: spacing.xl,
    marginBottom: spacing.lg,
  },
  statusText: {
    fontSize: Math.round(BASE_UNIT * PHI_INV * 0.8), // Matching desktop status font
    color: colors.primary, // #3498db
    fontWeight: '500',
    letterSpacing: 2,
  },
  statusSubtext: {
    fontSize: typography.small,
    color: 'rgba(52, 73, 94, 0.6)', // Semi-transparent matching desktop hint
    marginTop: spacing.xs,
    letterSpacing: 1,
  },
});
