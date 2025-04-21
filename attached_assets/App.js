import React, { useState, useRef, useEffect } from 'react';
import {
    View, Text, Button, TextInput, StyleSheet,
    ScrollView, ActivityIndicator, TouchableOpacity, Alert, Animated, Switch
} from 'react-native';
import { Ionicons } from '@expo/vector-icons';
import * as Animatable from 'react-native-animatable';
import { LinearGradient } from 'expo-linear-gradient';
import { useTheme, ThemeProvider } from './services/ThemeProvider'; // Import ThemeProvider
// If you use numeric ops in geometry:
import * as math from 'mathjs';

// Expo file system for CSV
import * as FileSystem from 'expo-file-system';
import * as Sharing from 'expo-sharing';

// QID engine & geometry classes
import { runQuantumDebug, exportCSV } from './services/QIDEngine';
import GeometricContainer from './services/GeometricContainer';
import LinearRegion from './services/LinearRegion';
import Shard from './services/shard';

// Components
import OscillatorChart from './services/OscillatorChartSVGCharts';
import AmplitudeHeatmap from './services/AmplitudeHeatmap';
import ContainerSchematic from './services/ContainerSchematic';


function AppContent() {
    const { colors } = useTheme(); // Get colors from theme provider
    const [numQubits, setNumQubits] = useState(3);
    const [inputData, setInputData] = useState('');

    // Loading and errors
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState(null);

    // Quantum debug log
    const [log, setLog] = useState(null);
    const [currentStep, setCurrentStep] = useState(0);

    // Container references
    const [container1, setContainer1] = useState(null);
    const [container2, setContainer2] = useState(null);
    // Toggles for oscillator & heatmap
    const [showOscillator, setShowOscillator] = useState(false);
    const [showHeatmap, setShowHeatmap] = useState(false);

    // Stress test results
    const [testResults, setTestResults] = useState(null);

    // Some fancy animations
    const fadeAnim = useRef(new Animated.Value(0)).current;
    const scrollRef = useRef();

    // Example geometry
    const containerMaterial = { youngsModulus: 1e9, density: 2700 };
    const container1Vertices = [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]];


    // RUN SIM
    const handleRunDebug = async () => {
        if (numQubits > 20) {
            Alert.alert("Too many qubits", "Keep it under 20 for demonstration.");
            return;
        }
        setLoading(true);
        setError(null);
        setLog(null);
        setTestResults(null);
        setShowOscillator(false);
        setShowHeatmap(false);

        try {
            // 1) Build containers
            const c1 = new GeometricContainer('container1', container1Vertices, [
                { rotation: { x: Math.PI / 4 } },
                { scale: { x: 1.2, y: 0.8 } },
                { translation: { x: 0.5, y: -0.5, z: 0.2 } }
            ], containerMaterial);
            c1.encodeData("101");
           c1.setBendAmount(0.2)
            c1.setInternalVibration(2)

           const c2 = new GeometricContainer('container2', [], [
                { rotation: { y: Math.PI / 2 } },
                { scale: { x: 0.7, y: 1.3 } },
                { translation: { x: -0.3, y: 0.7, z: 0.1 } }
            ], containerMaterial);
            c2.encodeData("010");

            const crescentVertices = c2.createCrescentVertices(1.5, 0.7);
            c2.updateVertices(crescentVertices)
            c2.setInternalVibration(1)


            // 2) Linear regions -> resonant frequencies
            const lr1 = new LinearRegion(container1Vertices);
           const lr2 = new LinearRegion(crescentVertices)
            c1.addLinearRegion(lr1);
            c2.addLinearRegion(lr2);

            c1.calculateResonantFrequencies(0.1);
            c2.calculateResonantFrequencies(0.1);

            // 3) Shard -> search
            const shard = new Shard([c1, c2]);
            const targetFreq = c1.resonantFrequencies[0];
            const searchRes = shard.search(targetFreq, 0.1);
            if (searchRes.length > 0) {
                Alert.alert("Resonance Found", `Matched: ${searchRes.map(cc => cc.id).join(', ')}`);
            } else {
                Alert.alert("No resonance", "None matched freq");
            }

            // 4) Quantum debug
            const logData = runQuantumDebug(inputData, numQubits);
            setLog(logData);
            setCurrentStep(0);

            // Keep container1/2 references for oscillator
            setContainer1(c1);
            setContainer2(c2)

            // Animate in
            Animated.timing(fadeAnim, {
                toValue: 1,
                duration: 1000,
                useNativeDriver: true
            }).start();

            scrollRef.current?.scrollTo({ y: 0, animated: true });
        } catch (err) {
            console.error("Simulation error:", err);
            setError("An error occurred during simulation");
        } finally {
            setLoading(false);
        }
    };

    // CSV EXPORT
    const handleExportCSV = async () => {
        try {
            await exportCSV();
        } catch (err) {
            console.error("Error exporting CSV:", err);
            setError("Error exporting CSV");
        }
    };

    // STRESS TEST
    const handleStressTest = async () => {
        if (numQubits > 12) {
            Alert.alert("Stress test too big", "Recommended max 12 qubits for stress test");
            return;
        }
        setLoading(true);
        setError(null);
        setLog(null);
        setTestResults(null);
        setShowOscillator(false);
        setShowHeatmap(false);

        try {
            const start = Date.now();
            const iterations = 1000;
            for (let i = 0; i < iterations; i++) {
                runQuantumDebug(inputData, numQubits);
            }
            const end = Date.now();
            setTestResults({
                count: iterations,
                executionTime: end - start,
                errorCount: 0
            });
            Alert.alert("Stress Test Complete", `Executed ${iterations} times in ${end - start} ms`);
        } catch (err) {
            console.error("Stress test error:", err);
            setError("Error during stress test");
        } finally {
            setLoading(false);
        }
    };

    // STEP NAV
    const handlePrevStep = () => {
        if (!log || !log.steps) return;
        if (currentStep > 0) {
            setCurrentStep(cur => cur - 1);
            scrollRef.current?.scrollTo({ y: 0, animated: true });
        }
    };
    const handleNextStep = () => {
        if (!log || !log.steps) return;
        if (currentStep < log.steps.length - 1) {
            setCurrentStep(cur => cur + 1);
            scrollRef.current?.scrollTo({ y: 0, animated: true });
        }
    };

    // MAP AMPLITUDE -> COLOR
    const getAmplitudeColor = (amp) => {
        let val = 0;
        const match = amp.match(/[-+]?[0-9]*\.?[0-9]+/);
        if (match) {
            val = Math.abs(parseFloat(match[0]));
        }
        const hue = Math.min(val * 300, 360);
        return `hsl(${hue},80%,60%)`;
    };

    // Toggle chart & heatmap
    const toggleOscillator = () => {
        setShowOscillator(!showOscillator);
    };
    const toggleHeatmap = () => {
        setShowHeatmap(!showHeatmap);
    };

    return (
        <LinearGradient
            colors={[colors.backgroundStart, colors.backgroundMiddle, colors.backgroundEnd]}
            style={styles.container}
        >
            <Text style={[styles.header, {color:colors.text}]}>Quantum Playground</Text>

            {/* Input row */}
            <View style={styles.row}>
                <Text style={[styles.label, {color:colors.text}]}># Qubits:</Text>
                <TextInput
                    style={[styles.input, {color: colors.text}]}
                    placeholder="1-20"
                    placeholderTextColor={colors.secondary}
                    keyboardType="number-pad"
                    value={String(numQubits)}
                    onChangeText={(txt) => {
                        const n = parseInt(txt, 10);
                        if (!isNaN(n) && n >= 1 && n <= 20) setNumQubits(n);
                    }}
                />
                <TouchableOpacity style={styles.runButton} onPress={handleRunDebug} disabled={loading}>
                    <Text style={[styles.runButtonText, {color: colors.text}]}>Run</Text>
                </TouchableOpacity>
            </View>

            {/* Another row for input data */}
            <View style={styles.row}>
                <TextInput
                    style={[styles.input, { flex: 1, color:colors.text}]}
                     placeholderTextColor={colors.secondary}
                    placeholder="Enter any input data"
                    value={inputData}
                    onChangeText={setInputData}
                />
            </View>

            {/* Export & Stress */}
            <View style={styles.buttonRow}>
                <TouchableOpacity style={styles.exportButton} onPress={handleExportCSV} disabled={loading || !log}>
                   <Text style={[styles.exportButtonText, {color: colors.text}]}>Export CSV</Text>
                </TouchableOpacity>
                <TouchableOpacity style={styles.stressButton} onPress={handleStressTest} disabled={loading}>
                  <Text style={[styles.stressButtonText, {color: colors.text}]}>Stress Test</Text>
                </TouchableOpacity>
            </View>

            {loading && <ActivityIndicator size="large" color={colors.primary} style={{ marginVertical: 10 }} />}
            {error && <Text style={styles.errorText}>{error}</Text>}

            <ScrollView ref={scrollRef} style={{ flex: 1, marginTop: 10 }}>
                {testResults && (
                    <Animatable.View style={[styles.card, styles.glassCard]} animation="fadeIn">
                        <Text style={[styles.title2, {color: colors.text}]}>Stress Test Results</Text>
                        <Text style={[styles.cardText, {color: colors.text}]}>Count: {testResults.count}</Text>
                        <Text style={[styles.cardText, {color: colors.text}]}>Exec Time: {testResults.executionTime}ms</Text>
                        <Text style={[styles.cardText, {color: colors.text}]}>Errors: {testResults.errorCount}</Text>
                    </Animatable.View>
                )}

                {log && log.steps && (
                    <Animated.View style={{ opacity: fadeAnim }}>
                        <Animatable.View style={[styles.card, styles.glassCard]} animation="fadeInUp">
                            <Text style={[styles.title2, {color: colors.text}]}>
                                Current Step: {currentStep} / {log.steps.length - 1}
                            </Text>

                            {/* Step nav */}
                            <View style={{ flexDirection: 'row', justifyContent: 'space-between', marginVertical: 10 }}>
                                <TouchableOpacity onPress={handlePrevStep} disabled={currentStep === 0}>
                                    <Ionicons name="arrow-back-circle-outline" size={40}
                                        color={currentStep === 0 ? 'gray' : colors.primary} />
                                </TouchableOpacity>
                                <TouchableOpacity onPress={handleNextStep} disabled={currentStep === log.steps.length - 1}>
                                    <Ionicons name="arrow-forward-circle-outline" size={40}
                                        color={currentStep === log.steps.length - 1 ? 'gray' : colors.primary} />
                                </TouchableOpacity>
                            </View>

                            {/* Step detail */}
                            {log.steps[currentStep] && (
                                <>
                                    <Text style={[styles.subTitle, {color: colors.text}]}>
                                        Step: {log.steps[currentStep].step}
                                    </Text>
                                    <View style={{ flexDirection: 'row', flexWrap: 'wrap', justifyContent: 'center' }}>
                                        {log.steps[currentStep].amplitude.map((amp, idx) => (
                                            <Animatable.View
                                                key={idx}
                                                animation="fadeInUp"
                                                duration={400}
                                                style={[
                                                    styles.ampBox,
                                                    { backgroundColor: getAmplitudeColor(amp) }
                                                ]}
                                            >
                                                <Text style={{ fontSize: 12, fontWeight: '600', color: '#fff' }}>Idx {idx}</Text>
                                                <Animatable.Text animation="pulse" style={{ fontSize: 12, color: '#fff' }}>
                                                    {amp}
                                                </Animatable.Text>
                                            </Animatable.View>
                                        ))}
                                    </View>
                                </>
                            )}
                        </Animatable.View>

                        <Text style={[styles.title2, { textAlign: 'center', color: colors.text }]}>
                            Final Measured State: {log.finalStateIndex}
                        </Text>

                        {/* Heatmap Toggle */}
                        {log.steps[currentStep] && (
                            <Animatable.View style={[styles.card, styles.glassCard, { marginTop: 8 }]} animation="fadeInUp">
                                <View style={styles.toggleRow}>
                                    <Text style={[styles.toggleText, {color: colors.text}]}>Show Amplitude Heatmap:</Text>
                                    <Switch value={showHeatmap} onValueChange={toggleHeatmap} trackColor={{ false: '#767577', true: colors.primary }}
                                        thumbColor={showHeatmap ? '#f4f3f4' : '#f4f3f4'} />
                                </View>
                                {showHeatmap && (
                                    <View style={{ marginTop: 10 }}>
                                        <AmplitudeHeatmap amplitudes={log.steps[currentStep].amplitude} />
                                    </View>
                                )}
                            </Animatable.View>
                        )}
                    </Animated.View>
                )}

                {/* Oscillator Toggle & Chart */}
                {container1 && container1.oscillator && (
                    <Animatable.View style={[styles.card, styles.glassCard, { marginTop: 8 }]} animation="fadeInUp">
                        <View style={styles.toggleRow}>
                            <Text style={[styles.toggleText, {color: colors.text}]}>Show Container1 Oscillator:</Text>
                            <Switch value={showOscillator} onValueChange={toggleOscillator} trackColor={{ false: '#767577', true: colors.primary }}
                                thumbColor={showOscillator ? '#f4f3f4' : '#f4f3f4'} />
                        </View>
                        {showOscillator && (
                            <View style={{ marginTop: 10 }}>
                                <Text style={{ fontWeight: 'bold', marginBottom: 5, color:colors.text }}>
                                    Frequency: {container1.oscillator.frequency.toFixed(2)} Hz
                                </Text>
                                <OscillatorChart oscillator={container1.oscillator} />
                            </View>
                        )}
                    </Animatable.View>
                )}

                {/* Container schematics */}
                {container1 && (
                    <Animatable.View style={[styles.card, styles.glassCard, { marginTop: 8 }]} animation="fadeInUp">
                        <Text style={[styles.title2, {color: colors.text}]}>Container 1 Schematic</Text>
                        <ContainerSchematic container={container1} size={150} />
                    </Animatable.View>
                )}
                {container2 && (
                    <Animatable.View style={[styles.card, styles.glassCard, { marginTop: 8 }]} animation="fadeInUp">
                        <Text style={[styles.title2, {color: colors.text}]}>Container 2 Schematic</Text>
                        <ContainerSchematic container={container2} size={150} />
                    </Animatable.View>
                )}
            </ScrollView>
        </LinearGradient>
    );
}

export default function App() {
    return (
        <ThemeProvider>
            <AppContent />
        </ThemeProvider>
    );
}

//--------------------------------- STYLES ---------------------------------//
const styles = StyleSheet.create({
    container: {
        flex: 1,
        paddingTop: 50,
        paddingHorizontal: 15,
    },
    header: {
        fontSize: 28,
        fontWeight: 'bold',
        textAlign: 'center',
        marginBottom: 20,
    },
    row: {
        flexDirection: 'row',
        alignItems: 'center',
        marginBottom: 15,
    },
    label: {
        marginRight: 10,
        fontWeight: '600',
    },
    input: {
        flex: 1,
        backgroundColor: '#2c3e50',
        borderWidth: 1,
        borderColor: '#555',
        borderRadius: 8,
        padding: 10,
        marginRight: 10
    },
    buttonRow: {
        flexDirection: 'row',
        justifyContent: 'space-between',
        marginBottom: 15,
    },
    errorText: {
        color: '#e74c3c',
        textAlign: 'center'
    },
    card: {
        padding: 15,
        borderRadius: 12,
        marginBottom: 15,
    },
    glassCard: {
        backgroundColor: 'rgba(255, 255, 255, 0.05)', // slight white
        borderWidth: 1,
        borderColor: 'rgba(255, 255, 255, 0.1)',
        shadowColor: '#000',
        shadowOffset: { width: 0, height: 2 },
        shadowOpacity: 0.3,
        shadowRadius: 3,
        elevation: 3, // shadow for android
        backdropFilter: 'blur(10px)', // blur for ios
    },
    title2: {
        fontSize: 18,
        fontWeight: 'bold',
        marginBottom: 8,
    },
    subTitle: {
        fontSize: 15,
        fontWeight: '600',
        marginBottom: 5,
        textAlign: 'center',
    },
    ampBox: {
        margin: 4,
        padding: 8,
        borderRadius: 8,
        minWidth: 70,
        alignItems: 'center',
        justifyContent: 'center',
    },
     toggleRow: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
  },
    toggleText: {
    },
   runButton: {
    backgroundColor: '#007AFF',
    padding: 12,
    borderRadius: 8,
    elevation: 3
   },
   runButtonText: {
     fontWeight: 'bold',
     fontSize: 16
   },
   exportButton: {
     backgroundColor: '#2ecc71',
     padding: 12,
     borderRadius: 8,
     elevation: 3
   },
   exportButtonText: {
     fontWeight: 'bold',
     fontSize: 16
   },
   stressButton: {
     backgroundColor: '#e67e22',
     padding: 12,
     borderRadius: 8,
     elevation: 3
   },
   stressButtonText: {
    fontWeight: 'bold',
    fontSize: 16
  },
    cardText: {
        fontSize: 14,
    },
});