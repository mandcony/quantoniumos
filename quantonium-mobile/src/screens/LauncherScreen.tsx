/**
 * Launcher Screen - QuantoniumOS Mobile
 * Main desktop/home screen with expandable circular arch
 * Matches desktop UI with Golden Ratio design and Q Logo
 */

import React, { useState, useEffect } from 'react';
import {
  View,
  Text,
  StyleSheet,
  TouchableOpacity,
  Dimensions,
  Animated,
} from 'react-native';
import { NativeStackNavigationProp } from '@react-navigation/native-stack';
import { RootStackParamList } from '../navigation/AppNavigator';
import QLogo from '../components/QLogo';
import AppIcon from '../components/AppIcon';
import { colors, spacing, typography, apps as appsList, PHI } from '../constants/DesignSystem';

type LauncherScreenNavigationProp = NativeStackNavigationProp<RootStackParamList, 'Launcher'>;

interface Props {
  navigation: LauncherScreenNavigationProp;
}

export default function LauncherScreen({ navigation }: Props) {
  const [currentTime, setCurrentTime] = useState('');
  const [isArchExpanded, setIsArchExpanded] = useState(false);
  const { width, height } = Dimensions.get('window');

  // Circular arch parameters (matching desktop)
  const centerX = width / 2;
  const centerY = height / 2;
  const archRadius = Math.min(width, height) * 0.35; // Responsive radius
  const buttonSize = 70;

  // Update time every minute (like desktop)
  useEffect(() => {
    const updateTime = () => {
      const now = new Date();
      const hours = now.getHours().toString().padStart(2, '0');
      const minutes = now.getMinutes().toString().padStart(2, '0');
      setCurrentTime(`${hours}:${minutes}`);
    };
    updateTime();
    const interval = setInterval(updateTime, 60000);
    return () => clearInterval(interval);
  }, []);

  const toggleArch = () => {
    setIsArchExpanded(!isArchExpanded);
  };

  // Calculate app positions in circular arch (matching desktop algorithm)
  const getAppPosition = (index: number, total: number) => {
    const angleStep = (Math.PI * 1.5) / (total - 1); // 270 degrees arc
    const startAngle = Math.PI * 1.75; // Start from top-left (315 degrees)
    const angle = startAngle - index * angleStep;

    return {
      x: centerX + archRadius * Math.cos(angle),
      y: centerY + archRadius * Math.sin(angle),
    };
  };

  return (
    <View style={styles.container}>
      {/* Time display (top right, like desktop) */}
      <View style={styles.timeContainer}>
        <Text style={styles.timeText}>{currentTime}</Text>
      </View>

      {/* Central Q Logo - Clickable to expand arch */}
      <TouchableOpacity
        style={[styles.logoContainer, { top: centerY - 60 }]}
        onPress={toggleArch}
        activeOpacity={0.8}
      >
        <QLogo size={120} color={colors.primary} />
        <Text style={styles.hint}>
          {isArchExpanded ? 'Tap Q to close' : 'Tap Q to show apps'}
        </Text>
      </TouchableOpacity>

      {/* Expandable Circular Arch */}
      {isArchExpanded && (
        <View style={styles.archContainer}>
          {/* Outer arch circle (subtle background) */}
          <View style={[styles.archBackground, {
            left: centerX - archRadius,
            top: centerY - archRadius,
            width: archRadius * 2,
            height: archRadius * 2,
          }]} />

          {/* Inner circle (aesthetic) */}
          <View style={[styles.innerCircle, {
            left: centerX - archRadius * 0.4,
            top: centerY - archRadius * 0.4,
            width: archRadius * 0.8,
            height: archRadius * 0.8,
          }]} />

          {/* Apps positioned in circular arch */}
          {appsList.map((app, index) => {
            const pos = getAppPosition(index, appsList.length);
            const displayName = app.name.split(' ').slice(-1)[0]; // Last word only

            return (
              <View
                key={app.id}
                style={[styles.appContainer, {
                  left: pos.x - buttonSize / 2,
                  top: pos.y - buttonSize / 2,
                }]}
              >
                <TouchableOpacity
                  style={[styles.appButton, { width: buttonSize, height: buttonSize }]}
                  onPress={() => {
                    setIsArchExpanded(false);
                    navigation.navigate(app.screen as keyof RootStackParamList);
                  }}
                >
                  <View style={styles.iconContainer}>
                    <AppIcon name={app.name} size={buttonSize * 0.6} color={colors.primary} />
                  </View>
                </TouchableOpacity>
                <Text style={styles.appLabel}>{displayName}</Text>
              </View>
            );
          })}
        </View>
      )}

      {/* QUANTONIUMOS branding (bottom, like desktop) */}
      <View style={styles.footer}>
        <Text style={styles.brandingText}>QUANTONIUMOS</Text>
        <Text style={styles.footerSubtext}>
          🚀 Symbolic Quantum-Inspired Computing
        </Text>
        <Text style={styles.patentText}>
          USPTO 19/169,399
        </Text>
      </View>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: colors.background, // White/light background like desktop
  },
  timeContainer: {
    position: 'absolute',
    top: spacing.lg,
    right: spacing.xl,
    zIndex: 20,
  },
  timeText: {
    fontSize: typography.body,
    color: colors.darkGray,
    fontWeight: '400',
    fontFamily: 'monospace',
  },
  logoContainer: {
    position: 'absolute',
    alignItems: 'center',
    alignSelf: 'center',
    zIndex: 15,
  },
  hint: {
    fontSize: typography.small,
    color: colors.gray,
    marginTop: spacing.md,
    opacity: 0.7,
  },
  archContainer: {
    position: 'absolute',
    width: '100%',
    height: '100%',
    zIndex: 10,
  },
  archBackground: {
    position: 'absolute',
    borderRadius: 1000,
    backgroundColor: 'rgba(255, 255, 255, 0.05)',
    borderWidth: 1,
    borderColor: 'rgba(52, 152, 219, 0.15)',
  },
  innerCircle: {
    position: 'absolute',
    borderRadius: 1000,
    backgroundColor: 'rgba(255, 255, 255, 0.03)',
    borderWidth: 1,
    borderColor: 'rgba(52, 152, 219, 0.1)',
  },
  appContainer: {
    position: 'absolute',
    alignItems: 'center',
  },
  appButton: {
    backgroundColor: 'rgba(255, 255, 255, 0.9)',
    borderRadius: 12,
    borderWidth: 1,
    borderColor: 'rgba(52, 152, 219, 0.3)',
    alignItems: 'center',
    justifyContent: 'center',
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.1,
    shadowRadius: 4,
    elevation: 3,
  },
  iconContainer: {
    alignItems: 'center',
    justifyContent: 'center',
  },
  appLabel: {
    fontSize: typography.micro,
    fontWeight: '600',
    color: colors.dark,
    textAlign: 'center',
    marginTop: spacing.xs,
    maxWidth: 80,
  },
  footer: {
    position: 'absolute',
    bottom: spacing.xl,
    alignSelf: 'center',
    alignItems: 'center',
    paddingHorizontal: spacing.lg,
  },
  brandingText: {
    fontSize: typography.subtitle,
    fontWeight: '600',
    color: colors.primary,
    letterSpacing: 1.2,
    marginBottom: spacing.xs,
  },
  footerSubtext: {
    fontSize: typography.small,
    color: colors.gray,
    textAlign: 'center',
    marginBottom: spacing.xs,
  },
  patentText: {
    fontSize: typography.micro,
    color: colors.lightGray,
    textAlign: 'center',
  },
});
