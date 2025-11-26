/**
 * Q Logo Component - QuantoniumOS Mobile
 * Replicates the desktop Q logo with Golden Ratio proportions
 */

import React from 'react';
import { View, StyleSheet } from 'react-native';
import Svg, { Circle, Line } from 'react-native-svg';

interface QLogoProps {
  size?: number;
  color?: string;
}

export default function QLogo({ size = 80, color = '#3498db' }: QLogoProps) {
  // Golden Ratio constants (matching desktop)
  const phi = 1.618033988749895;
  const phi_inv = 1 / phi;

  // Calculate dimensions based on Golden Ratio
  const baseUnit = size / 5; // Scale to requested size
  const outerRadius = baseUnit * (phi * phi); // φ² scaling
  const innerRadius = outerRadius * phi_inv; // 1/φ scaling
  const strokeWidth = Math.max(2, baseUnit * 0.15); // Precise stroke

  // Calculate diagonal line for Q
  const lineLength = outerRadius * phi_inv;
  const lineAngle = Math.PI / 4; // 45 degrees
  const lineDx = Math.cos(lineAngle) * lineLength;
  const lineDy = Math.sin(lineAngle) * lineLength;
  const offset = innerRadius * phi_inv * 0.5;

  // Center point
  const center = size / 2;

  return (
    <View style={[styles.container, { width: size, height: size }]}>
      <Svg width={size} height={size}>
        {/* Outer circle */}
        <Circle
          cx={center}
          cy={center}
          r={outerRadius}
          stroke={color}
          strokeWidth={strokeWidth}
          fill="none"
          strokeLinecap="round"
        />

        {/* Inner circle */}
        <Circle
          cx={center}
          cy={center}
          r={innerRadius}
          stroke={color}
          strokeWidth={strokeWidth}
          fill="none"
          strokeLinecap="round"
        />

        {/* Diagonal dash to make it a Q */}
        <Line
          x1={center + offset}
          y1={center + offset}
          x2={center + offset + lineDx}
          y2={center + offset + lineDy}
          stroke={color}
          strokeWidth={strokeWidth}
          strokeLinecap="round"
        />
      </Svg>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    justifyContent: 'center',
    alignItems: 'center',
  },
});
