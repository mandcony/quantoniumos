/**
 * ChipViewer3DScreen - QuantoniumOS Mobile
 * True 3D RFTPU chip visualization using expo-gl + Three.js
 * Matches desktop RFTPU3DChipDissect component with full 3D rendering
 * 
 * Copyright (C) 2025 Luis M. Minier / QuantoniumOS
 * USPTO Patent Application #19/169,399
 */

import React, { useState, useRef, useEffect, useCallback } from 'react';
import {
  View,
  Text,
  StyleSheet,
  TouchableOpacity,
  ScrollView,
  Dimensions,
  Platform,
  SafeAreaView,
  StatusBar,
} from 'react-native';
import { GLView, ExpoWebGLRenderingContext } from 'expo-gl';
import * as THREE from 'three';
import { colors, spacing, typography } from '../constants/DesignSystem';

const { width: SCREEN_WIDTH, height: SCREEN_HEIGHT } = Dimensions.get('window');

// ===============================================
// CHIP CONFIGURATIONS: WebFPGA vs ASIC TLV
// ===============================================
interface ChipConfig {
  id: string;
  name: string;
  subtitle: string;
  status: string;
  statusColor: string;
  process: string;
  package: string;
  clockActual?: string;
  clockTarget?: string;
  tiles: { rows: number; cols: number };
  kernelModes: number;
  luts?: string;
  lutPercent?: string;
  nocTopology?: string;
  specs: Record<string, string>;
  color: number;
  warning?: string;
}

const CHIP_CONFIGS: Record<'webfpga' | 'asic', ChipConfig> = {
  webfpga: {
    id: 'webfpga',
    name: 'WebFPGA iCE40 HX8K',
    subtitle: 'Synthesized & Proven',
    status: 'REAL HARDWARE',
    statusColor: '#00ff00',
    process: 'Lattice iCE40 HX8K',
    package: 'CT256 (256-pin)',
    clockActual: '21.9 MHz',
    tiles: { rows: 1, cols: 1 },
    kernelModes: 12,
    luts: '7,680 LUTs',
    lutPercent: '35.68%',
    specs: {
      'LUTs': '7,680 (35.68% used)',
      'Max Freq': '21.9 MHz Achieved',
      'Architecture': 'Single RFT Kernel',
      'Kernel Modes': '12 validated modes',
      'ROM Entries': '768 (12×64 coeffs)',
      'Unitarity': '< 1e-13 error',
    },
    color: 0x3498db,
  },
  asic: {
    id: 'asic',
    name: 'RFTPU 64-Tile ASIC',
    subtitle: 'Architecture Design (TLV)',
    status: 'DESIGN ONLY',
    statusColor: '#ffaa00',
    process: 'TSMC 7nm FinFET',
    package: 'BGA-800 flip-chip',
    clockTarget: '950 MHz',
    tiles: { rows: 8, cols: 8 },
    kernelModes: 12,
    nocTopology: '8×8 Mesh',
    specs: {
      'Tiles': '64 (8×8 mesh)',
      'Target Freq': '950 MHz (tiles)',
      'NoC Freq': '1.2 GHz',
      'Process': '7nm FinFET',
      'Peak Perf': '2.39 TOPS',
      'Efficiency': '291 GOPS/W',
    },
    color: 0x9b59b6,
    warning: '⚠️ Design projections, NOT silicon',
  },
};

// WebFPGA kernel modes
const WEBFPGA_KERNEL_MODES = [
  { id: 0, name: 'RFT-Golden', desc: 'Golden ratio resonance (PRIMARY)', unitarity: '6.12e-15' },
  { id: 1, name: 'RFT-Fibonacci', desc: 'Fibonacci frequency structure', unitarity: '1.09e-13' },
  { id: 2, name: 'RFT-Harmonic', desc: 'Natural harmonic overtones', unitarity: '<1e-13' },
  { id: 3, name: 'RFT-Geometric', desc: 'Self-similar φ^i frequencies', unitarity: '<1e-13' },
  { id: 4, name: 'RFT-Beating', desc: 'Golden ratio interference', unitarity: '<1e-13' },
  { id: 5, name: 'RFT-Phyllotaxis', desc: 'Golden angle 137.5° (bio)', unitarity: '<1e-13' },
  { id: 6, name: 'RFT-Cascade', desc: 'H3 DCT+RFT blend', unitarity: '<1e-13' },
  { id: 7, name: 'RFT-Hybrid-DCT', desc: 'Split DCT/RFT basis', unitarity: '<1e-13' },
  { id: 8, name: 'RFT-Manifold', desc: 'Manifold projection (+47.9dB)', unitarity: '<1e-13' },
  { id: 9, name: 'RFT-Euler', desc: 'Spherical geodesic', unitarity: '<1e-13' },
  { id: 10, name: 'RFT-PhaseCoh', desc: 'Phase-space coherence', unitarity: '<1e-13' },
  { id: 11, name: 'RFT-Entropy', desc: 'Entropy-modulated chaos', unitarity: '<1e-13' },
];

interface TileActivity {
  row: number;
  col: number;
  active: boolean;
  load: number;
  temperature: number;
}

export default function ChipViewer3DScreen() {
  const [activeChip, setActiveChip] = useState<'webfpga' | 'asic'>('webfpga');
  const [selectedKernel, setSelectedKernel] = useState(0);
  const [showKernelPicker, setShowKernelPicker] = useState(false);
  const [selectedTile, setSelectedTile] = useState<TileActivity | null>(null);
  const [autoRotate, setAutoRotate] = useState(true);
  const [showSpecs, setShowSpecs] = useState(false);
  
  // Three.js refs
  const rendererRef = useRef<THREE.WebGLRenderer | null>(null);
  const sceneRef = useRef<THREE.Scene | null>(null);
  const cameraRef = useRef<THREE.PerspectiveCamera | null>(null);
  const chipGroupRef = useRef<THREE.Group | null>(null);
  const tilesRef = useRef<THREE.Mesh[]>([]);
  const frameIdRef = useRef<number>(0);
  const rotationRef = useRef({ x: 0.3, y: 0 });
  const glRef = useRef<ExpoWebGLRenderingContext | null>(null);

  const config = CHIP_CONFIGS[activeChip];

  // Generate tile activity data
  const [tileActivity, setTileActivity] = useState<TileActivity[][]>([]);

  useEffect(() => {
    const { rows, cols } = config.tiles;
    const newTiles: TileActivity[][] = [];
    for (let r = 0; r < rows; r++) {
      const row: TileActivity[] = [];
      for (let c = 0; c < cols; c++) {
        row.push({
          row: r,
          col: c,
          active: Math.random() > 0.3,
          load: Math.random() * 100,
          temperature: 45 + Math.random() * 40,
        });
      }
      newTiles.push(row);
    }
    setTileActivity(newTiles);

    // Animate tile activity
    const interval = setInterval(() => {
      setTileActivity(prev => prev.map(row =>
        row.map(tile => ({
          ...tile,
          load: Math.max(0, Math.min(100, tile.load + (Math.random() - 0.5) * 15)),
          temperature: Math.max(45, Math.min(95, tile.temperature + (Math.random() - 0.5) * 5)),
          active: Math.random() > 0.2,
        }))
      ));
    }, 800);

    return () => clearInterval(interval);
  }, [activeChip]);

  // Build 3D chip model
  const buildChip = useCallback((group: THREE.Group, chipConfig: ChipConfig) => {
    // Clear existing
    while (group.children.length > 0) {
      group.remove(group.children[0]);
    }
    tilesRef.current = [];

    const isASIC = chipConfig.id === 'asic';
    const { rows, cols } = chipConfig.tiles;

    // Package substrate
    const substrateGeo = new THREE.BoxGeometry(4, 0.3, 4);
    const substrateMat = new THREE.MeshPhongMaterial({
      color: 0x1a1a2e,
      shininess: 30,
    });
    const substrate = new THREE.Mesh(substrateGeo, substrateMat);
    substrate.position.y = -0.35;
    group.add(substrate);

    // Die
    const dieGeo = new THREE.BoxGeometry(3, 0.2, 3);
    const dieMat = new THREE.MeshPhongMaterial({
      color: 0x2c3e50,
      shininess: 60,
    });
    const die = new THREE.Mesh(dieGeo, dieMat);
    die.position.y = -0.1;
    group.add(die);

    // Create tiles grid
    const tileSize = isASIC ? 0.3 : 2.5;
    const tileGap = isASIC ? 0.05 : 0;
    const gridSize = (tileSize + tileGap) * Math.max(rows, cols);
    const startX = -gridSize / 2 + tileSize / 2;
    const startZ = -gridSize / 2 + tileSize / 2;

    for (let r = 0; r < rows; r++) {
      for (let c = 0; c < cols; c++) {
        const tileGeo = new THREE.BoxGeometry(tileSize * 0.9, 0.15, tileSize * 0.9);
        const tileMat = new THREE.MeshPhongMaterial({
          color: chipConfig.color,
          shininess: 80,
          transparent: true,
          opacity: 0.9,
        });
        const tile = new THREE.Mesh(tileGeo, tileMat);
        
        tile.position.x = startX + c * (tileSize + tileGap);
        tile.position.y = 0.08;
        tile.position.z = startZ + r * (tileSize + tileGap);
        
        tile.userData = { row: r, col: c };
        tilesRef.current.push(tile);
        group.add(tile);

        // Add glow effect for active tiles
        if (isASIC) {
          const glowGeo = new THREE.BoxGeometry(tileSize * 0.85, 0.02, tileSize * 0.85);
          const glowMat = new THREE.MeshBasicMaterial({
            color: 0x00ff88,
            transparent: true,
            opacity: 0.3,
          });
          const glow = new THREE.Mesh(glowGeo, glowMat);
          glow.position.copy(tile.position);
          glow.position.y += 0.1;
          tile.userData.glow = glow;
          group.add(glow);
        }
      }
    }

    // Heat spreader (IHS) - semi-transparent
    const ihsGeo = new THREE.BoxGeometry(3.2, 0.15, 3.2);
    const ihsMat = new THREE.MeshPhongMaterial({
      color: 0xc0c0c0,
      shininess: 100,
      transparent: true,
      opacity: 0.4,
    });
    const ihs = new THREE.Mesh(ihsGeo, ihsMat);
    ihs.position.y = 0.25;
    group.add(ihs);

    // BGA balls (bottom)
    const ballRadius = 0.04;
    const ballRows = isASIC ? 12 : 8;
    const ballCols = isASIC ? 12 : 8;
    const ballSpacing = 0.28;
    const ballStartX = -(ballCols - 1) * ballSpacing / 2;
    const ballStartZ = -(ballRows - 1) * ballSpacing / 2;

    const ballGeo = new THREE.SphereGeometry(ballRadius, 8, 8);
    const ballMat = new THREE.MeshPhongMaterial({
      color: 0xffd700,
      shininess: 100,
    });

    for (let br = 0; br < ballRows; br++) {
      for (let bc = 0; bc < ballCols; bc++) {
        const ball = new THREE.Mesh(ballGeo, ballMat);
        ball.position.x = ballStartX + bc * ballSpacing;
        ball.position.y = -0.55;
        ball.position.z = ballStartZ + br * ballSpacing;
        group.add(ball);
      }
    }

    // Package edge pins
    const pinGeo = new THREE.BoxGeometry(0.08, 0.25, 0.02);
    const pinMat = new THREE.MeshPhongMaterial({ color: 0xffd700 });
    const pinCount = isASIC ? 20 : 12;
    
    for (let i = 0; i < pinCount; i++) {
      const offset = -1.8 + (3.6 / (pinCount - 1)) * i;
      
      // Top edge
      const pinTop = new THREE.Mesh(pinGeo, pinMat);
      pinTop.position.set(offset, -0.2, -2.1);
      group.add(pinTop);
      
      // Bottom edge
      const pinBottom = new THREE.Mesh(pinGeo, pinMat);
      pinBottom.position.set(offset, -0.2, 2.1);
      group.add(pinBottom);
      
      // Left edge
      const pinLeft = new THREE.Mesh(pinGeo, pinMat);
      pinLeft.rotation.y = Math.PI / 2;
      pinLeft.position.set(-2.1, -0.2, offset);
      group.add(pinLeft);
      
      // Right edge
      const pinRight = new THREE.Mesh(pinGeo, pinMat);
      pinRight.rotation.y = Math.PI / 2;
      pinRight.position.set(2.1, -0.2, offset);
      group.add(pinRight);
    }

    // Q Logo on top
    const qGeo = new THREE.TorusGeometry(0.2, 0.04, 8, 24);
    const qMat = new THREE.MeshPhongMaterial({
      color: 0x3498db,
      emissive: 0x3498db,
      emissiveIntensity: 0.3,
    });
    const qTorus = new THREE.Mesh(qGeo, qMat);
    qTorus.position.y = 0.35;
    qTorus.rotation.x = Math.PI / 2;
    group.add(qTorus);

    // Q tail
    const tailGeo = new THREE.BoxGeometry(0.04, 0.02, 0.15);
    const tail = new THREE.Mesh(tailGeo, qMat);
    tail.position.set(0.15, 0.35, 0.1);
    tail.rotation.y = -Math.PI / 4;
    group.add(tail);
  }, []);

  // Initialize Three.js scene
  const onContextCreate = useCallback((gl: ExpoWebGLRenderingContext) => {
    glRef.current = gl;
    
    // Create renderer
    const renderer = new THREE.WebGLRenderer({
      // @ts-ignore - expo-gl context compatibility
      canvas: {
        width: gl.drawingBufferWidth,
        height: gl.drawingBufferHeight,
        style: {},
        addEventListener: () => {},
        removeEventListener: () => {},
        clientHeight: gl.drawingBufferHeight,
        getContext: () => gl,
      } as any,
      context: gl as any,
    });
    renderer.setSize(gl.drawingBufferWidth, gl.drawingBufferHeight);
    renderer.setPixelRatio(1);
    renderer.setClearColor(0x0f0c29, 1);
    rendererRef.current = renderer;

    // Create scene
    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0x0f0c29);
    sceneRef.current = scene;

    // Create camera
    const camera = new THREE.PerspectiveCamera(
      45,
      gl.drawingBufferWidth / gl.drawingBufferHeight,
      0.1,
      1000
    );
    camera.position.set(0, 3, 6);
    camera.lookAt(0, 0, 0);
    cameraRef.current = camera;

    // Add lights
    const ambientLight = new THREE.AmbientLight(0x404040, 0.6);
    scene.add(ambientLight);

    const directionalLight = new THREE.DirectionalLight(0xffffff, 0.8);
    directionalLight.position.set(5, 10, 7.5);
    scene.add(directionalLight);

    const pointLight = new THREE.PointLight(0x3498db, 0.5, 20);
    pointLight.position.set(-5, 5, 5);
    scene.add(pointLight);

    // Create chip group
    const chipGroup = new THREE.Group();
    scene.add(chipGroup);
    chipGroupRef.current = chipGroup;

    // Build initial chip
    buildChip(chipGroup, config);

    // Animation loop
    const animate = () => {
      frameIdRef.current = requestAnimationFrame(animate);

      if (chipGroupRef.current) {
        if (autoRotate) {
          rotationRef.current.y += 0.005;
        }
        chipGroupRef.current.rotation.y = rotationRef.current.y;
        chipGroupRef.current.rotation.x = rotationRef.current.x;
      }

      renderer.render(scene, camera);
      gl.endFrameEXP();
    };

    animate();
  }, [config, buildChip, autoRotate]);

  // Update tiles based on activity
  useEffect(() => {
    if (!tilesRef.current.length || !tileActivity.length) return;

    tilesRef.current.forEach((tile) => {
      const { row, col } = tile.userData;
      if (row !== undefined && col !== undefined && tileActivity[row]?.[col]) {
        const activity = tileActivity[row][col];
        const mat = tile.material as THREE.MeshPhongMaterial;
        
        // Color based on load/temperature
        const hue = (1 - activity.load / 100) * 0.3; // Green to red
        mat.color.setHSL(hue, 0.8, 0.5);
        mat.emissive.setHSL(hue, 0.8, activity.active ? 0.2 : 0);
        
        // Update glow if exists
        if (tile.userData.glow) {
          const glowMat = tile.userData.glow.material as THREE.MeshBasicMaterial;
          glowMat.opacity = activity.active ? 0.4 : 0.1;
        }
      }
    });
  }, [tileActivity]);

  // Rebuild chip when config changes
  useEffect(() => {
    if (chipGroupRef.current) {
      buildChip(chipGroupRef.current, config);
    }
  }, [activeChip, buildChip, config]);

  // Cleanup
  useEffect(() => {
    return () => {
      if (frameIdRef.current) {
        cancelAnimationFrame(frameIdRef.current);
      }
    };
  }, []);

  return (
    <SafeAreaView style={styles.container}>
      <StatusBar barStyle="light-content" backgroundColor="#0f0c29" />
      
      {/* Header */}
      <View style={styles.header}>
        <View style={styles.headerLeft}>
          <Text style={styles.title}>RFTPU 3D</Text>
          <View style={[styles.statusBadge, { borderColor: config.statusColor }]}>
            <View style={[styles.statusDot, { backgroundColor: config.statusColor }]} />
            <Text style={[styles.statusText, { color: config.statusColor }]}>
              {config.status}
            </Text>
          </View>
        </View>
        <TouchableOpacity
          style={styles.autoRotateBtn}
          onPress={() => setAutoRotate(!autoRotate)}
        >
          <Text style={styles.autoRotateBtnText}>
            {autoRotate ? '⏸ Pause' : '▶ Rotate'}
          </Text>
        </TouchableOpacity>
      </View>

      {/* Chip Selector */}
      <View style={styles.chipSelector}>
        <TouchableOpacity
          style={[styles.chipBtn, activeChip === 'webfpga' && styles.chipBtnActive]}
          onPress={() => setActiveChip('webfpga')}
        >
          <Text style={[styles.chipBtnText, activeChip === 'webfpga' && styles.chipBtnTextActive]}>
            🟢 WebFPGA
          </Text>
          <Text style={styles.chipBtnSubtext}>REAL</Text>
        </TouchableOpacity>
        <TouchableOpacity
          style={[styles.chipBtn, activeChip === 'asic' && styles.chipBtnActive]}
          onPress={() => setActiveChip('asic')}
        >
          <Text style={[styles.chipBtnText, activeChip === 'asic' && styles.chipBtnTextActive]}>
            🟠 ASIC TLV
          </Text>
          <Text style={styles.chipBtnSubtext}>DESIGN</Text>
        </TouchableOpacity>
      </View>

      {/* Kernel Mode Selector (WebFPGA only) */}
      {activeChip === 'webfpga' && (
        <TouchableOpacity
          style={styles.kernelSelector}
          onPress={() => setShowKernelPicker(!showKernelPicker)}
        >
          <Text style={styles.kernelLabel}>Kernel:</Text>
          <Text style={styles.kernelValue}>{WEBFPGA_KERNEL_MODES[selectedKernel].name}</Text>
          <Text style={styles.kernelArrow}>{showKernelPicker ? '▲' : '▼'}</Text>
        </TouchableOpacity>
      )}

      {/* Kernel Picker Dropdown */}
      {showKernelPicker && (
        <ScrollView style={styles.kernelPicker} nestedScrollEnabled>
          {WEBFPGA_KERNEL_MODES.map((mode) => (
            <TouchableOpacity
              key={mode.id}
              style={[styles.kernelOption, selectedKernel === mode.id && styles.kernelOptionActive]}
              onPress={() => {
                setSelectedKernel(mode.id);
                setShowKernelPicker(false);
              }}
            >
              <Text style={styles.kernelOptionName}>{mode.name}</Text>
              <Text style={styles.kernelOptionDesc}>{mode.desc}</Text>
              <Text style={styles.kernelUnitarity}>η: {mode.unitarity}</Text>
            </TouchableOpacity>
          ))}
        </ScrollView>
      )}

      {/* 3D GL View */}
      <View style={styles.glContainer}>
        <GLView
          style={styles.glView}
          onContextCreate={onContextCreate}
        />
        <Text style={styles.glHint}>3D View • Auto-rotating</Text>
      </View>

      {/* Chip Info */}
      <View style={styles.chipInfo}>
        <Text style={styles.chipName}>{config.name}</Text>
        <Text style={styles.chipSubtitle}>{config.subtitle}</Text>
        {config.warning && (
          <Text style={styles.chipWarning}>{config.warning}</Text>
        )}
      </View>

      {/* Specs Toggle */}
      <TouchableOpacity
        style={styles.specsToggle}
        onPress={() => setShowSpecs(!showSpecs)}
      >
        <Text style={styles.specsToggleText}>
          {showSpecs ? '▼ Hide Specs' : '▶ Show Specs'}
        </Text>
      </TouchableOpacity>

      {/* Specs Panel */}
      {showSpecs && (
        <ScrollView style={styles.specsPanel}>
          {Object.entries(config.specs).map(([key, value]) => (
            <View key={key} style={styles.specRow}>
              <Text style={styles.specKey}>{key}</Text>
              <Text style={styles.specValue}>{String(value)}</Text>
            </View>
          ))}
        </ScrollView>
      )}

      {/* Footer */}
      <View style={styles.footer}>
        <Text style={styles.footerText}>USPTO 19/169,399 • DOI: 10.5281/zenodo.17822056</Text>
      </View>
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#0f0c29',
    paddingTop: Platform.OS === 'android' ? StatusBar.currentHeight : 0,
  },
  header: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    paddingHorizontal: spacing.md,
    paddingVertical: spacing.sm,
  },
  headerLeft: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: spacing.sm,
  },
  title: {
    fontSize: typography.title,
    fontWeight: 'bold',
    color: '#fff',
  },
  statusBadge: {
    flexDirection: 'row',
    alignItems: 'center',
    paddingHorizontal: spacing.sm,
    paddingVertical: 4,
    borderRadius: 12,
    borderWidth: 1,
  },
  statusDot: {
    width: 6,
    height: 6,
    borderRadius: 3,
    marginRight: 4,
  },
  statusText: {
    fontSize: 10,
    fontWeight: '700',
  },
  autoRotateBtn: {
    backgroundColor: 'rgba(52, 152, 219, 0.3)',
    paddingHorizontal: spacing.sm,
    paddingVertical: 6,
    borderRadius: 8,
  },
  autoRotateBtnText: {
    color: '#3498db',
    fontSize: 12,
    fontWeight: '600',
  },
  chipSelector: {
    flexDirection: 'row',
    marginHorizontal: spacing.md,
    marginBottom: spacing.sm,
    gap: spacing.sm,
  },
  chipBtn: {
    flex: 1,
    paddingVertical: spacing.sm,
    alignItems: 'center',
    backgroundColor: 'rgba(255,255,255,0.05)',
    borderRadius: 8,
    borderWidth: 1,
    borderColor: 'transparent',
  },
  chipBtnActive: {
    backgroundColor: 'rgba(52, 152, 219, 0.2)',
    borderColor: '#3498db',
  },
  chipBtnText: {
    color: '#888',
    fontSize: 13,
    fontWeight: '600',
  },
  chipBtnTextActive: {
    color: '#fff',
  },
  chipBtnSubtext: {
    color: '#666',
    fontSize: 9,
    marginTop: 2,
  },
  kernelSelector: {
    flexDirection: 'row',
    alignItems: 'center',
    marginHorizontal: spacing.md,
    marginBottom: spacing.sm,
    backgroundColor: 'rgba(255,255,255,0.05)',
    padding: spacing.sm,
    borderRadius: 8,
  },
  kernelLabel: {
    color: '#888',
    fontSize: 12,
    marginRight: spacing.sm,
  },
  kernelValue: {
    flex: 1,
    color: '#3498db',
    fontSize: 12,
    fontWeight: '600',
  },
  kernelArrow: {
    color: '#666',
    fontSize: 10,
  },
  kernelPicker: {
    maxHeight: 180,
    marginHorizontal: spacing.md,
    marginBottom: spacing.sm,
    backgroundColor: 'rgba(20,20,40,0.95)',
    borderRadius: 8,
  },
  kernelOption: {
    padding: spacing.sm,
    borderBottomWidth: 1,
    borderBottomColor: 'rgba(255,255,255,0.05)',
  },
  kernelOptionActive: {
    backgroundColor: 'rgba(52, 152, 219, 0.2)',
  },
  kernelOptionName: {
    color: '#fff',
    fontSize: 12,
    fontWeight: '600',
  },
  kernelOptionDesc: {
    color: '#888',
    fontSize: 10,
    marginTop: 2,
  },
  kernelUnitarity: {
    color: '#00ff88',
    fontSize: 9,
    marginTop: 2,
  },
  glContainer: {
    flex: 1,
    marginHorizontal: spacing.md,
    marginBottom: spacing.sm,
    borderRadius: 12,
    overflow: 'hidden',
    backgroundColor: '#0f0c29',
    borderWidth: 1,
    borderColor: 'rgba(52,152,219,0.3)',
  },
  glView: {
    flex: 1,
  },
  glHint: {
    position: 'absolute',
    bottom: 8,
    alignSelf: 'center',
    color: 'rgba(255,255,255,0.4)',
    fontSize: 10,
  },
  chipInfo: {
    alignItems: 'center',
    paddingVertical: spacing.xs,
  },
  chipName: {
    color: '#fff',
    fontSize: 14,
    fontWeight: '600',
  },
  chipSubtitle: {
    color: '#888',
    fontSize: 11,
  },
  chipWarning: {
    color: '#ffaa00',
    fontSize: 9,
    marginTop: 2,
  },
  specsToggle: {
    marginHorizontal: spacing.md,
    paddingVertical: spacing.xs,
  },
  specsToggleText: {
    color: '#3498db',
    fontSize: 12,
    fontWeight: '600',
  },
  specsPanel: {
    maxHeight: 120,
    marginHorizontal: spacing.md,
    marginBottom: spacing.sm,
    backgroundColor: 'rgba(255,255,255,0.03)',
    borderRadius: 8,
    padding: spacing.sm,
  },
  specRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    paddingVertical: 3,
  },
  specKey: {
    color: '#888',
    fontSize: 11,
  },
  specValue: {
    color: '#3498db',
    fontSize: 11,
    fontWeight: '600',
  },
  footer: {
    alignItems: 'center',
    paddingVertical: spacing.sm,
    borderTopWidth: 1,
    borderTopColor: 'rgba(255,255,255,0.05)',
  },
  footerText: {
    color: '#444',
    fontSize: 9,
  },
});
