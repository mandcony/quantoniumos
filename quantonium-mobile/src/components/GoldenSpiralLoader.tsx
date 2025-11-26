/**
 * Golden Spiral Loader - Exact 1:1 match with desktop QuantoniumOS
 * Radial spiral animation using φ (Golden Ratio) proportions
 */

import React, { useEffect, useRef } from 'react';
import { View, StyleSheet, Animated, Dimensions } from 'react-native';
import Svg, { Path } from 'react-native-svg';

const PHI = 1.618033988749895;

interface GoldenSpiralLoaderProps {
  size?: number;
  onFinished?: () => void;
}

export default function GoldenSpiralLoader({ 
  size = Math.min(Dimensions.get('window').width, Dimensions.get('window').height) * 0.55,
  onFinished 
}: GoldenSpiralLoaderProps) {
  const opacity = useRef(new Animated.Value(0)).current;
  const animationTime = useRef(0);
  const animationRef = useRef<number | undefined>(undefined);

  const SPIRAL_COUNT = 8;
  const POINTS_PER_ARM = 60;
  const ARM_DELAY_MS = 80;
  const FADE_IN_DURATION = 1500;
  const HOLD_DURATION = 1000;
  const FADE_OUT_DURATION = 1000;
  const TOTAL_DURATION = FADE_IN_DURATION + HOLD_DURATION + FADE_OUT_DURATION;

  const maxRadius = Math.min(220, Math.max(120, size * 0.33));
  const strokeWidth = Math.max(2, size * 0.007);

  // Build spiral paths (matching desktop algorithm exactly)
  const buildSpiralPaths = (): string[] => {
    const paths: string[] = [];
    
    for (let arm = 0; arm < SPIRAL_COUNT; arm++) {
      let pathData = '';
      
      for (let idx = 0; idx < POINTS_PER_ARM; idx++) {
        const t = idx / Math.max(1, POINTS_PER_ARM - 1);
        const angle = arm * (2 * Math.PI / SPIRAL_COUNT) + t * Math.PI * 4;
        const radius = Math.pow(PHI, t * 0.2) * t * maxRadius;
        const x = radius * Math.cos(angle);
        const y = radius * Math.sin(angle);
        
        if (idx === 0) {
          pathData += `M ${x} ${y}`;
        } else {
          pathData += ` L ${x} ${y}`;
        }
      }
      
      paths.push(pathData);
    }
    
    return paths;
  };

  const spiralPaths = buildSpiralPaths();

  // Calculate alpha for each arm (matching desktop timing)
  const getArmAlpha = (armIndex: number, elapsedMs: number): number => {
    const delay = armIndex * ARM_DELAY_MS;
    const adjusted = elapsedMs - delay;
    
    if (adjusted <= 0) return 0;
    
    if (adjusted < FADE_IN_DURATION) {
      return adjusted / FADE_IN_DURATION;
    }
    
    if (adjusted < FADE_IN_DURATION + HOLD_DURATION) {
      const pulse = 0.5 + 0.3 * Math.sin(elapsedMs * 0.003 + armIndex * 0.5);
      return pulse;
    }
    
    if (adjusted < TOTAL_DURATION) {
      const fadeProgress = (adjusted - FADE_IN_DURATION - HOLD_DURATION) / FADE_OUT_DURATION;
      const base = Math.max(0, 1 - fadeProgress);
      const pulse = 0.3 + 0.2 * Math.sin(elapsedMs * 0.003 + armIndex * 0.5);
      return base * pulse;
    }
    
    return 0;
  };

  useEffect(() => {
    // Fade in animation
    Animated.timing(opacity, {
      toValue: 1,
      duration: 600,
      useNativeDriver: true,
    }).start();

    // Animation loop
    const startTime = Date.now();
    let fadeOutStarted = false;

    const animate = () => {
      const elapsed = Date.now() - startTime;
      animationTime.current = elapsed;

      // Trigger fade out
      if (!fadeOutStarted && elapsed > TOTAL_DURATION + 500) {
        fadeOutStarted = true;
        Animated.timing(opacity, {
          toValue: 0,
          duration: 700,
          useNativeDriver: true,
        }).start(() => {
          if (onFinished) onFinished();
        });
      }

      // Continue animation until fade out completes
      if (elapsed < TOTAL_DURATION + 1500) {
        animationRef.current = requestAnimationFrame(animate);
      }
    };

    animationRef.current = requestAnimationFrame(animate);

    return () => {
      if (animationRef.current) {
        cancelAnimationFrame(animationRef.current);
      }
    };
  }, []);

  return (
    <Animated.View style={[styles.container, { opacity }]}>
      <Svg width={size} height={size} viewBox={`${-maxRadius - strokeWidth} ${-maxRadius - strokeWidth} ${(maxRadius + strokeWidth) * 2} ${(maxRadius + strokeWidth) * 2}`}>
        {spiralPaths.map((path, armIndex) => {
          const alpha = getArmAlpha(armIndex, animationTime.current);
          return (
            <Path
              key={armIndex}
              d={path}
              stroke={`rgba(36, 113, 163, ${Math.max(0, Math.min(1, alpha))})`}
              strokeWidth={strokeWidth}
              strokeLinecap="round"
              strokeLinejoin="round"
              fill="none"
            />
          );
        })}
      </Svg>
    </Animated.View>
  );
}

const styles = StyleSheet.create({
  container: {
    justifyContent: 'center',
    alignItems: 'center',
  },
});
