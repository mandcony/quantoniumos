// services/QIDEngine.js
import * as FileSystem from 'expo-file-system';
import * as Sharing from 'expo-sharing';
import { MultiQubitState } from './multiQubitState';
import GeometricContainer from './GeometricContainer';
import LinearRegion from './LinearRegion'; // Make sure this is imported
import Shard from './shard'; // Make sure this is imported
import QuantumSearch from './QuantumSearch'; // Make sure this is imported!
import { Alert } from 'react-native'; // Import Alert from react-native

// We'll keep a local data log in memory
let localDataLog = [];

export function runQuantumDebug(inputData = "N/A", numQubits = 3) {
    const timestamp = new Date().toISOString();
    const stepLogs = [];

    // 1. Quantum simulation
    // Create multi-qubit state
    const state = new MultiQubitState(numQubits);
    stepLogs.push({ step: 'Initial', amplitude: state.getAmplitudeInfo() });

    // Example: Apply a Hadamard gate (H) on qubit 0
    const hGate = [
        [
            (amp) => ({ expr: `(1/sqrt(2))*${amp.expr}` }),
            (amp) => ({ expr: `(1/sqrt(2))*${amp.expr}` })
        ],
        [
            (amp) => ({ expr: `(1/sqrt(2))*${amp.expr}` }),
            (amp) => ({ expr: `(-1/sqrt(2))*${amp.expr}` })
        ],
    ];
    state.applySingleQubitGate(hGate, 0);
    stepLogs.push({ step: 'Apply H to qubit 0', amplitude: state.getAmplitudeInfo() });

    // If there is more than one qubit, apply a CNOT from qubit 0 to qubit 1
    if (numQubits > 1) {
        state.applyTwoQubitGate('CNOT', 0, 1);
        stepLogs.push({ step: 'Apply CNOT(0->1)', amplitude: state.getAmplitudeInfo() });
    }

    // Perform measurement
    const finalIndex = state.measureAll();
    stepLogs.push({ step: `Measured => ${finalIndex}`, amplitude: state.getAmplitudeInfo() });

    // Retrieve the final quantum state amplitude info for use in container transformations.
    // (Depending on your needs you may want to use an earlier state.)
    const quantumState = state.getAmplitudeInfo();

    // 2. Build geometry containers
    const containerMaterial = { youngsModulus: 1e9, density: 2700 };
    const container1Vertices = [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]];
    const container2Vertices = [[2, 2, 0], [3, 2, 0], [3, 3, 0], [2, 3, 0]]; // Vertices for container 2

    // Build container 1 with transformations and encode data (using inputData)
    const c1 = new GeometricContainer('container1', container1Vertices, [
        { rotation: { x: Math.PI / 4 } },
        { scale: { x: 1.2, y: 0.8 } },
        { translation: { x: 0.5, y: -0.5, z: 0.2 } }
    ], containerMaterial);
    c1.encodeData(inputData);

    // Build container 2 with transformations and encode different data ("010")
    const c2 = new GeometricContainer('container2', container2Vertices, [
        { rotation: { y: Math.PI / 2 } },
        { scale: { x: 0.7, y: 1.3 } },
        { translation: { x: -0.3, y: 0.7, z: 0.1 } }
    ], containerMaterial);
    c2.encodeData("010");

    // 3. Define linear regions and calculate resonant frequencies for each container
    const lr1 = new LinearRegion(container1Vertices);
    const lr2 = new LinearRegion(container2Vertices);
    c1.addLinearRegion(lr1);
    c2.addLinearRegion(lr2);

    c1.calculateResonantFrequencies(0.1);
    c2.calculateResonantFrequencies(0.1);

    // 4. Apply quantum state transformations to each container.
    // This will adjust their vertices based on the quantum state.
    c1.applyAllTransformations(quantumState);
    c2.applyAllTransformations(quantumState);

    // 5. Use a Shard to group containers and perform a quantum search
    const shard = new Shard([c1, c2]);
    const targetFreq = c1.encodedFrequency;  // Target frequency is taken from container1

    const quantumSearch = new QuantumSearch();
    const searchRes = quantumSearch.search([c1, c2], targetFreq);
    if (searchRes && searchRes.id === 'container1') {
        Alert.alert("Quantum Resonance Found", `Matched: ${searchRes.id}`);
    } else {
        Alert.alert("Quantum No Resonance", "None matched the target frequency");
    }

    // 6. Log the simulation result and return the entry
    const entry = {
        entryID: localDataLog.length + 1,
        timestamp,
        inputData,
        steps: stepLogs,
        finalStateIndex: finalIndex
    };
    localDataLog.push(entry);
    return entry;
}

export async function exportCSV() {
    // Build a CSV string from the local data log
    const csvString = localDataLog
        .map(e => `${e.entryID},${e.timestamp},${e.inputData},${e.finalStateIndex}`)
        .join('\n');

    if (!csvString) {
        Alert.alert("No data to export");
        return;
    }
    try {
        const fileUri = `${FileSystem.documentDirectory}quantum_debug_results.csv`;
        await FileSystem.writeAsStringAsync(fileUri, csvString, { encoding: FileSystem.EncodingType.UTF8 });
        await Sharing.shareAsync(fileUri);
    } catch (err) {
        console.error("Error exporting CSV:", err);
        Alert.alert("Error exporting CSV", "Check console for details.");
    }
}
