/**
 * Q Splash Screen - QuantoniumOS Mobile
 * Exact 1:1 match with desktop boot sequence using GoldenSpiralLoader
 */

import React, { useEffect, useRef, useState } from 'react';
import { Animated, StyleSheet, Text, View } from 'react-native';
import QLogo from './QLogo';
import GoldenSpiralLoader from './GoldenSpiralLoader';
import { colors, typography } from '../constants/DesignSystem';

export default function QSplashScreen() {
  const fadeAnim = useRef(new Animated.Value(0)).current;
  const [showLoader, setShowLoader] = useState(true);
  const [showLogo, setShowLogo] = useState(false);

  useEffect(() => {
    // Fade in container
    Animated.timing(fadeAnim, {
      toValue: 1,
      duration: 600,
      useNativeDriver: true,
    }).start();
  }, []);

  const handleLoaderFinished = () => {
    setShowLoader(false);
    setShowLogo(true);
  };

  return (
    <Animated.View style={[styles.container, { opacity: fadeAnim }]}>
      {/* White background matching desktop */}
      <View style={StyleSheet.absoluteFillObject} />

      {/* Golden Spiral Loader (matching desktop intro) */}
      {showLoader && (
        <View style={styles.loaderContainer}>
          <GoldenSpiralLoader onFinished={handleLoaderFinished} />
        </View>
      )}

      {/* Q Logo appears after spiral completes */}
      {showLogo && (
        <View style={styles.logoContainer}>
          <QLogo size={160} color={colors.primary} />
          <Text style={styles.title}>QUANTONIUMOS</Text>
          <Text style={styles.subtitle}>Click Q to show apps</Text>
        </View>
      )}
    </Animated.View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    alignItems: 'center',
    justifyContent: 'center',
    backgroundColor: colors.background, // White like desktop
  },
  loaderContainer: {
    alignItems: 'center',
    justifyContent: 'center',
  },
  logoContainer: {
    alignItems: 'center',
    justifyContent: 'center',
  },
  title: {
    marginTop: 28,
    fontSize: typography.subtitle,
    color: colors.primary, // #3498db matching desktop
    fontWeight: '500',
    letterSpacing: 2,
  },
  subtitle: {
    marginTop: 12,
    fontSize: typography.small,
    color: 'rgba(52, 73, 94, 0.6)', // Semi-transparent matching desktop hint
    letterSpacing: 0.5,
  },
});
