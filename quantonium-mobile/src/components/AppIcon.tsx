/**
 * App Icon Component - Simple SVG-based icons
 */

import React from 'react';
import { View } from 'react-native';
import Svg, { Path, Circle, Rect, G } from 'react-native-svg';

interface AppIconProps {
  name: string;
  size?: number;
  color?: string;
}

export default function AppIcon({ name, size = 48, color = '#ffffff' }: AppIconProps) {
  const iconMap: Record<string, React.ReactElement> = {
    'RFT Validation Suite': (
      <Svg width={size} height={size} viewBox="0 0 24 24">
        <Path d="M9 16.17L4.83 12l-1.42 1.41L9 19 21 7l-1.41-1.41z" fill={color} />
      </Svg>
    ),
    'AI Chat': (
      <Svg width={size} height={size} viewBox="0 0 24 24">
        <Path d="M20 2H4c-1.1 0-2 .9-2 2v18l4-4h14c1.1 0 2-.9 2-2V4c0-1.1-.9-2-2-2zm-3 12H7v-2h10v2zm0-3H7V9h10v2zm0-3H7V6h10v2z" fill={color} />
      </Svg>
    ),
    'Quantum Simulator': (
      <Svg width={size} height={size} viewBox="0 0 24 24">
        <Circle cx="12" cy="12" r="3" fill="none" stroke={color} strokeWidth="2" />
        <Circle cx="12" cy="12" r="8" fill="none" stroke={color} strokeWidth="1.5" />
        <Path d="M12 4L12 1M12 23L12 20M20 12L23 12M1 12L4 12" stroke={color} strokeWidth="1.5" />
      </Svg>
    ),
    'Quantum Cryptography': (
      <Svg width={size} height={size} viewBox="0 0 24 24">
        <Path d="M18 8h-1V6c0-2.76-2.24-5-5-5S7 3.24 7 6v2H6c-1.1 0-2 .9-2 2v10c0 1.1.9 2 2 2h12c1.1 0 2-.9 2-2V10c0-1.1-.9-2-2-2zM9 6c0-1.66 1.34-3 3-3s3 1.34 3 3v2H9V6z" fill={color} />
      </Svg>
    ),
    'System Monitor': (
      <Svg width={size} height={size} viewBox="0 0 24 24">
        <Path d="M3 13h8V3H3v10zm0 8h8v-6H3v6zm10 0h8V11h-8v10zm0-18v6h8V3h-8z" fill={color} />
      </Svg>
    ),
    'Q-Notes': (
      <Svg width={size} height={size} viewBox="0 0 24 24">
        <Path d="M14 2H6c-1.1 0-2 .9-2 2v16c0 1.1.9 2 2 2h12c1.1 0 2-.9 2-2V8l-6-6zM6 20V4h7v5h5v11H6z" fill={color} />
      </Svg>
    ),
    'Q-Vault': (
      <Svg width={size} height={size} viewBox="0 0 24 24">
        <Rect x="3" y="7" width="18" height="14" rx="2" fill="none" stroke={color} strokeWidth="2" />
        <Path d="M7 7V5c0-1.66 1.34-3 3-3h4c1.66 0 3 1.34 3 3v2" fill="none" stroke={color} strokeWidth="2" />
        <Circle cx="12" cy="14" r="2" fill={color} />
      </Svg>
    ),
    'Structural Health': (
      <Svg width={size} height={size} viewBox="0 0 24 24">
        <Path d="M4 20h16v2H4z" fill={color} />
        <Path d="M6 18h3l2-6 2 6h3l-4-12z" fill={color} />
        <Path d="M11 8h2l1-3h-4z" fill={color} opacity={0.7} />
      </Svg>
    ),
  };

  return iconMap[name] || (
    <Svg width={size} height={size} viewBox="0 0 24 24">
      <Circle cx="12" cy="12" r="10" fill={color} />
    </Svg>
  );
}
