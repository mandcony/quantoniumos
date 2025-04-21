/**
 * Quantonium OS - Qubit Step Visualizer
 * 
 * FRONTEND ONLY - VISUALIZATION COMPONENT
 * 
 * This is a frontend visualization that connects to Quantonium's proprietary quantum engines.
 * All actual quantum computation happens in the protected backend.
 * This visualization merely represents quantum states for educational purposes.
 * 
 * NOTICE: The proprietary Quantonium quantum algorithms, 150-qubit capabilities,
 * and scientific innovations are implemented in protected backend modules and  
 * are not represented in this client-side code.
 */

// Complex number class
class Complex {
    constructor(real, imag) {
        this.real = real;
        this.imag = imag;
    }
    
    add(other) {
        return new Complex(this.real + other.real, this.imag + other.imag);
    }
    
    multiply(other) {
        const re = this.real * other.real - this.imag * other.imag;
        const im = this.real * other.imag + this.imag * other.real;
        return new Complex(re, im);
    }
    
    toString() {
        const sign = this.imag >= 0 ? '+' : '';
        return `${this.real}${sign}${this.imag}i`;
    }
    
    static fromPolar(r, theta) {
        return new Complex(r * Math.cos(theta), r * Math.sin(theta));
    }
    
    magnitude() {
        return Math.sqrt(this.real * this.real + this.imag * this.imag);
    }
    
    phase() {
        return Math.atan2(this.imag, this.real);
    }
}

// Enhanced Multi Qubit State class 
class EnhancedMultiQubitState {
    constructor(numQubits) {
        if (!Number.isInteger(numQubits) || numQubits <= 0) {
            throw new Error("Number of qubits must be a positive integer.");
        }

        this.numQubits = numQubits;
        this.stateTensor = this.initializeTensor();
        this.entanglementMap = {}; // Track qubit correlations (simulated)
    }

    initializeTensor() {
        const dimension = 2 ** this.numQubits;
        const tensor = new Array(dimension);
        
        // Initialize to |0...0⟩ state
        for (let i = 0; i < dimension; i++) {
            tensor[i] = { real: 0, imag: 0 };
        }
        tensor[0] = { real: 1, imag: 0 }; // |0...0⟩ state
        
        return tensor;
    }

    applyGate(gateMatrix) {
        if (!Array.isArray(gateMatrix)) {
            throw new Error("Gate matrix must be an array.");
        }

        const matrixDimension = gateMatrix.length;
        if ((matrixDimension & (matrixDimension - 1)) !== 0 || matrixDimension !== this.stateTensor.length) {
            throw new Error("Gate matrix must be a square matrix with dimensions that are a power of 2 and equal the state tensor length.");
        }

        // Matrix-vector product implementation, handling complex numbers
        const newStateTensor = new Array(this.stateTensor.length);

        for (let i = 0; i < this.stateTensor.length; i++) {
            let newReal = 0;
            let newImag = 0;
            for (let j = 0; j < this.stateTensor.length; j++) {
                // Complex multiplication: (a + bi) * (c + di) = (ac - bd) + (ad + bc)i
                const gateElement = gateMatrix[i][j];
                if(typeof gateElement !== 'object' || !('real' in gateElement) || !('imag' in gateElement)) {
                    throw new Error("Gate matrix element is not a complex number object {real:number, imag:number}.");
                }
                const stateElement = this.stateTensor[j];

                newReal += (gateElement.real * stateElement.real) - (gateElement.imag * stateElement.imag);
                newImag += (gateElement.real * stateElement.imag) + (gateElement.imag * stateElement.real);
            }
            newStateTensor[i] = {real: newReal, imag: newImag};
        }

        this.stateTensor = newStateTensor;
    }

    entangle(qubitA, qubitB) {
        if (!Number.isInteger(qubitA) || !Number.isInteger(qubitB) || 
            qubitA < 0 || qubitA >= this.numQubits || 
            qubitB < 0 || qubitB >= this.numQubits || 
            qubitA === qubitB) {
            throw new Error("Invalid qubit indices for entanglement.");
        }

        const key = `${qubitA}-${qubitB}`;
        this.entanglementMap[key] = true;
    }

    measure() {
        // Calculate probabilities (magnitude squared of complex amplitudes)
        let probabilities = this.stateTensor.map(amp => amp.real * amp.real + amp.imag * amp.imag);

        // Normalize the probabilities to ensure they sum to 1.
        const sumOfProbabilities = probabilities.reduce((sum, probability) => sum + probability, 0);
        probabilities = probabilities.map(probability => probability / sumOfProbabilities);

        let randomNumber = Math.random();
        let cumulativeProbability = 0;

        for (let i = 0; i < probabilities.length; i++) {
            cumulativeProbability += probabilities[i];
            if (randomNumber <= cumulativeProbability) {
                return i.toString(2).padStart(this.numQubits, '0');
            }
        }

        // Should not happen, but return the last state if there are floating point precision issues.
        return (probabilities.length - 1).toString(2).padStart(this.numQubits, '0');
    }
    
    // Get probabilities for all states
    getProbabilities() {
        // Calculate probabilities (magnitude squared of complex amplitudes)
        let probabilities = this.stateTensor.map(amp => amp.real * amp.real + amp.imag * amp.imag);

        // Normalize the probabilities to ensure they sum to 1.
        const sumOfProbabilities = probabilities.reduce((sum, probability) => sum + probability, 0);
        if (sumOfProbabilities > 0) {
            probabilities = probabilities.map(probability => probability / sumOfProbabilities);
        }
        
        return probabilities.map((prob, index) => {
            return {
                state: index.toString(2).padStart(this.numQubits, '0'),
                probability: prob
            };
        });
    }
    
    // Get the Bloch sphere representation for a single qubit
    getBlochSphereCoordinates(qubitIndex) {
        if (qubitIndex >= this.numQubits) {
            throw new Error("Qubit index out of range");
        }
        
        // For a general state, calculate the reduced density matrix for this qubit
        // and extract Bloch sphere coordinates (simplified approach)
        
        // For demonstration, we'll use a simplified calculation that works for 
        // basic states but isn't fully general
        
        // Calculate probabilities of |0⟩ and |1⟩ for this qubit
        const prob0 = this.getProbabilityOfQubitInState(qubitIndex, 0);
        const prob1 = 1 - prob0;
        
        // Simple approximation for Bloch coordinates
        // In a real system this would use the full density matrix
        const theta = 2 * Math.acos(Math.sqrt(prob0));
        const phi = Math.random() * 2 * Math.PI; // We can't recover phi from just probabilities
        
        // Convert to Cartesian coordinates
        const x = Math.sin(theta) * Math.cos(phi);
        const y = Math.sin(theta) * Math.sin(phi);
        const z = Math.cos(theta);
        
        return { x, y, z };
    }
    
    // Calculate probability of a specific qubit being in state 0 or 1
    getProbabilityOfQubitInState(qubitIndex, bitValue) {
        let totalProb = 0;
        const mask = 1 << qubitIndex;
        
        for (let i = 0; i < this.stateTensor.length; i++) {
            const bitIsSet = (i & mask) !== 0;
            const stateMatches = (bitValue === 1 && bitIsSet) || (bitValue === 0 && !bitIsSet);
            
            if (stateMatches) {
                const amp = this.stateTensor[i];
                totalProb += amp.real * amp.real + amp.imag * amp.imag;
            }
        }
        
        return totalProb;
    }
}

// Geometric Quantum Mapper - for mapping quantum states to geometric spaces
class GeometricQuantumMapper {
    constructor(dimensions) {
        if (!Number.isInteger(dimensions) || dimensions <= 0) {
            throw new Error("Dimensions must be a positive integer.");
        }

        this.dimensions = dimensions;
        this.space = this.initializeGeometricSpace();
    }

    initializeGeometricSpace() {
        const space = new Array(this.dimensions);
        for (let i = 0; i < this.dimensions; i++) {
            space[i] = Math.random() * 2 - 1;
        }
        return space;
    }

    encodeState(qubitState) {
        if (typeof qubitState !== 'string' || !/^[01]+$/.test(qubitState)) {
            throw new Error("Qubit state must be a string containing only '0' and '1' characters.");
        }

        const encodedState = new Array(qubitState.length);
        for (let i = 0; i < qubitState.length; i++) {
            encodedState[i] = (qubitState[i] === "1" ? 1 : -1);
        }
        return encodedState;
    }

    transformState(stateVector) {
        if (!Array.isArray(stateVector)) {
            throw new Error("State vector must be an array.");
        }

        const transformedVector = new Array(stateVector.length);
        for (let i = 0; i < stateVector.length; i++) {
            const coord = stateVector[i];
            if (typeof coord !== 'number') {
                throw new Error("State vector must contain only numbers.");
            }
            transformedVector[i] = Math.sin(coord * Math.PI);
        }
        return transformedVector;
    }
}

// Global variables
let scene, camera, renderer, controls;
let qubitSpheres = [];
let qubitVectors = [];
let currentState = [];
let stateVector = {};
let animationSpeed = 5;
let qubitCount = 2;
let currentAlgorithm = null;
let algorithmStep = 1;
let isAnimating = false;

// Quantum system
let quantumState = new EnhancedMultiQubitState(qubitCount);
let geometricMapper = new GeometricQuantumMapper(3);

// Constants for quantum gates
const GATES = {
    h: { name: "Hadamard", matrix: [[1/Math.sqrt(2), 1/Math.sqrt(2)], [1/Math.sqrt(2), -1/Math.sqrt(2)]] },
    x: { name: "Pauli-X", matrix: [[0, 1], [1, 0]] },
    y: { name: "Pauli-Y", matrix: [[0, -1i], [1i, 0]] },
    z: { name: "Pauli-Z", matrix: [[1, 0], [0, -1]] },
    cnot: { name: "CNOT", controlled: true },
    swap: { name: "SWAP", multiQubit: true }
};

// Algorithm definitions
const ALGORITHMS = {
    bell: {
        name: "Bell State",
        description: "Creates a maximally entangled state between two qubits, demonstrating quantum entanglement.",
        steps: [
            { name: "Initialize", gate: null, target: null, description: "Initialize qubits to |00⟩" },
            { name: "Hadamard", gate: "h", target: 0, description: "Apply Hadamard to first qubit" },
            { name: "CNOT", gate: "cnot", control: 0, target: 1, description: "Apply CNOT with control=0, target=1" },
            { name: "Measure", gate: null, target: null, description: "Result: entangled Bell state" }
        ]
    },
    grover: {
        name: "Grover's Search (2-qubit)",
        description: "Demonstrates quantum search algorithm to find a marked item in an unsorted database with quadratic speedup.",
        steps: [
            { name: "Initialize", gate: null, target: null, description: "Initialize qubits to |00⟩" },
            { name: "Hadamard All", gate: "h", target: "all", description: "Apply Hadamard to all qubits" },
            { name: "Oracle", gate: "custom", target: null, description: "Apply quantum oracle (marks solution)" },
            { name: "Diffusion", gate: "custom", target: null, description: "Apply diffusion operator" },
            { name: "Measure", gate: null, target: null, description: "Result: amplified solution state" }
        ]
    },
    deutsch: {
        name: "Deutsch Algorithm",
        description: "Determines if a function is constant or balanced with only one function evaluation.",
        steps: [
            { name: "Initialize", gate: null, target: null, description: "Initialize qubits |0⟩|1⟩" },
            { name: "Hadamard All", gate: "h", target: "all", description: "Apply Hadamard to all qubits" },
            { name: "Oracle", gate: "custom", target: null, description: "Apply quantum oracle (function)" },
            { name: "Hadamard", gate: "h", target: 0, description: "Apply Hadamard to first qubit" },
            { name: "Measure", gate: null, target: null, description: "Result: |0⟩ if constant, |1⟩ if balanced" }
        ]
    },
    teleport: {
        name: "Quantum Teleportation",
        description: "Transfers a quantum state from one location to another using entanglement and classical communication.",
        steps: [
            { name: "Initialize", gate: null, target: null, description: "Initialize 3 qubits |000⟩" },
            { name: "Prepare", gate: "custom", target: 0, description: "Prepare state to teleport" },
            { name: "Bell Pair", gate: "custom", target: null, description: "Create Bell pair between qubits 1 and 2" },
            { name: "Entangle", gate: "cnot", control: 0, target: 1, description: "Entangle qubit 0 with Bell pair" },
            { name: "Hadamard", gate: "h", target: 0, description: "Apply Hadamard to qubit 0" },
            { name: "Measure", gate: "custom", target: null, description: "Measure qubits 0 and 1" },
            { name: "Correction", gate: "custom", target: 2, description: "Apply conditional gates to qubit 2" },
            { name: "Result", gate: null, target: null, description: "Qubit 2 now has the original state of qubit 0" }
        ]
    }
};

// DOM elements (initialized in DOMContentLoaded)
let elements = {};

// Initialize the application when DOM is ready
document.addEventListener('DOMContentLoaded', () => {
    initElements();
    setupEventListeners();
    initVisualization();
    resetQuantumState();
    updateUIForQubitCount();
    updateAlgorithmDescription();
});

// Initialize DOM element references
function initElements() {
    elements = {
        // Mode buttons
        simulationModeBtn: document.getElementById('simulation-mode-btn'),
        algorithmModeBtn: document.getElementById('algorithm-mode-btn'),
        
        // Panels
        simulationPanel: document.getElementById('simulation-panel'),
        algorithmPanel: document.getElementById('algorithm-panel'),
        
        // Simulation controls
        qubitCount: document.getElementById('qubit-count'),
        quantumGate: document.getElementById('quantum-gate'),
        targetQubit: document.getElementById('target-qubit'),
        controlQubit: document.getElementById('control-qubit'),
        controlQubitContainer: document.getElementById('control-qubit-container'),
        applyGateBtn: document.getElementById('apply-gate-btn'),
        resetStateBtn: document.getElementById('reset-state-btn'),
        
        // State display
        stateVector: document.getElementById('state-vector'),
        stateProbability: document.getElementById('state-probability'),
        
        // Algorithm controls
        quantumAlgorithm: document.getElementById('quantum-algorithm'),
        algorithmDescription: document.getElementById('algorithm-description'),
        runAlgorithmBtn: document.getElementById('run-algorithm-btn'),
        stepAlgorithmBtn: document.getElementById('step-algorithm-btn'),
        resetAlgorithmBtn: document.getElementById('reset-algorithm-btn'),
        
        // Steps
        steps: document.querySelectorAll('.step'),
        
        // Visualization
        blochSphereContainer: document.getElementById('bloch-sphere-container'),
        viewMode: document.getElementById('view-mode'),
        animationSpeed: document.getElementById('animation-speed'),
        
        // Status
        statusMessage: document.getElementById('status-message')
    };
}

// Set up event listeners
function setupEventListeners() {
    // Mode switching
    elements.simulationModeBtn.addEventListener('click', () => switchMode('simulation'));
    elements.algorithmModeBtn.addEventListener('click', () => switchMode('algorithm'));
    elements.quantumGridBtn.addEventListener('click', () => switchMode('quantum-grid'));
    
    // Simulation controls
    elements.qubitCount.addEventListener('change', handleQubitCountChange);
    elements.quantumGate.addEventListener('change', handleGateChange);
    elements.applyGateBtn.addEventListener('click', applyGate);
    elements.resetStateBtn.addEventListener('click', resetQuantumState);
    
    // Algorithm controls
    elements.quantumAlgorithm.addEventListener('change', updateAlgorithmDescription);
    elements.runAlgorithmBtn.addEventListener('click', runAlgorithm);
    elements.stepAlgorithmBtn.addEventListener('click', stepAlgorithm);
    elements.resetAlgorithmBtn.addEventListener('click', resetAlgorithm);
    
    // Visualization controls
    elements.viewMode.addEventListener('change', switchVisualizationMode);
    elements.animationSpeed.addEventListener('input', (e) => {
        animationSpeed = parseInt(e.target.value);
    });
}

// Switch between simulation, algorithm, and quantum grid modes
function switchMode(mode) {
    // Reset all buttons and panels
    elements.simulationModeBtn.classList.remove('active');
    elements.algorithmModeBtn.classList.remove('active');
    elements.quantumGridBtn.classList.remove('active');
    
    elements.simulationPanel.classList.remove('active');
    elements.algorithmPanel.classList.remove('active');
    elements.quantumGridPanel.classList.remove('active');
    
    // Set active state based on the selected mode
    if (mode === 'simulation') {
        elements.simulationModeBtn.classList.add('active');
        elements.simulationPanel.classList.add('active');
    } else if (mode === 'algorithm') {
        elements.algorithmModeBtn.classList.add('active');
        elements.algorithmPanel.classList.add('active');
    } else if (mode === 'quantum-grid') {
        elements.quantumGridBtn.classList.add('active');
        elements.quantumGridPanel.classList.add('active');
        // Initialize the quantum grid if not already done
        initializeQuantumGrid();
    }
    
    // Reset the visualization for simulation and algorithm modes
    if (mode !== 'quantum-grid') {
        resetQuantumState();
    }
}

// Handle qubit count change
function handleQubitCountChange() {
    qubitCount = parseInt(elements.qubitCount.value);
    updateUIForQubitCount();
    resetQuantumState();
}

// Update UI based on qubit count
function updateUIForQubitCount() {
    // Update target qubit dropdown
    elements.targetQubit.innerHTML = '';
    for (let i = 0; i < qubitCount; i++) {
        const option = document.createElement('option');
        option.value = i;
        option.textContent = `Qubit ${i}`;
        elements.targetQubit.appendChild(option);
    }
    
    // Update control qubit dropdown
    elements.controlQubit.innerHTML = '';
    for (let i = 0; i < qubitCount; i++) {
        const option = document.createElement('option');
        option.value = i;
        option.textContent = `Qubit ${i}`;
        elements.controlQubit.appendChild(option);
    }
    
    // Show/hide control qubit based on gate selection
    handleGateChange();
}

// Handle gate selection change
function handleGateChange() {
    const selectedGate = elements.quantumGate.value;
    const gate = GATES[selectedGate];
    
    // Show/hide control qubit for controlled gates
    if (gate && gate.controlled) {
        elements.controlQubitContainer.style.display = 'flex';
    } else {
        elements.controlQubitContainer.style.display = 'none';
    }
}

// Apply quantum gate to the current state
function applyGate() {
    if (isAnimating) return;
    
    const gateType = elements.quantumGate.value;
    const targetQubit = parseInt(elements.targetQubit.value);
    const controlQubit = parseInt(elements.controlQubit.value);
    
    if (gateType === 'cnot' && targetQubit === controlQubit) {
        updateStatus('Error: Control and target qubits must be different');
        return;
    }
    
    updateStatus(`Applying ${GATES[gateType].name} gate to qubit ${targetQubit}...`);
    
    // Simulate applying the gate
    isAnimating = true;
    
    // Call into the quantum engine here (using setTimeout to simulate for now)
    setTimeout(() => {
        applyGateToState(gateType, targetQubit, controlQubit);
        updateQuantumState();
        isAnimating = false;
    }, 500);
}

// Apply gate to the quantum state 
// This is a frontend function that sends commands to the protected backend
function applyGateToState(gateType, targetQubit, controlQubit) {
    // Use both our quantum state model and the simplified currentState for visualization
    
    try {
        // First, update the frontend visualization model (simplified)
        switch(gateType) {
            case 'h': // Hadamard
                // Update the simplified model for visualization support
                if (currentState[targetQubit] === 0) {
                    currentState[targetQubit] = 0.5; // Representing |+⟩ state
                } else if (currentState[targetQubit] === 1) {
                    currentState[targetQubit] = -0.5; // Representing |-⟩ state
                } else if (currentState[targetQubit] === 0.5) {
                    currentState[targetQubit] = Math.random() < 0.5 ? 0 : 1; // Collapse
                } else if (currentState[targetQubit] === -0.5) {
                    currentState[targetQubit] = Math.random() < 0.5 ? 0 : 1; // Collapse
                }
                break;
                
            case 'x': // Pauli-X (NOT gate)
                // Update the simplified model for visualization
                if (currentState[targetQubit] === 0) {
                    currentState[targetQubit] = 1;
                } else if (currentState[targetQubit] === 1) {
                    currentState[targetQubit] = 0;
                }
                break;
                
            case 'z': // Pauli-Z
                // Phase flip isn't visible in the computational basis,
                // but affects the Bloch sphere visualization
                updateStatus(`Applied Z gate to qubit ${targetQubit} (phase change)`);
                break;
                
            case 'cnot': // Controlled-NOT
                if (currentState[controlQubit] === 1) {
                    // Flip the target qubit if control is |1⟩
                    if (currentState[targetQubit] === 0) {
                        currentState[targetQubit] = 1;
                    } else if (currentState[targetQubit] === 1) {
                        currentState[targetQubit] = 0;
                    }
                }
                
                // Mark these qubits as entangled in our model
                quantumState.entangle(controlQubit, targetQubit);
                updateStatus(`Applied CNOT gate with control=${controlQubit}, target=${targetQubit}`);
                break;
                
            case 'swap': // SWAP gate
                // Swap states of two qubits in the simplified model
                const temp = currentState[targetQubit];
                currentState[targetQubit] = currentState[controlQubit];
                currentState[controlQubit] = temp;
                
                updateStatus(`Swapped qubits ${targetQubit} and ${controlQubit}`);
                break;
        }
        
        // In parallel, send the quantum operation to the backend engine
        // This is where we connect to your protected proprietary engine
        sendQuantumOperationToBackend(gateType, targetQubit, controlQubit);
        
    } catch (error) {
        console.error("Error applying gate:", error);
        updateStatus(`Error applying gate: ${error.message}`);
    }
    
    // Update the visualization
    updateQuantumState();
    updateBlochSphereVisualization();
}

// Function to communicate with the protected backend quantum engine
async function sendQuantumOperationToBackend(gateType, targetQubit, controlQubit = null) {
    try {
        // Create a circuit definition for the backend
        const circuitDefinition = {
            gates: [
                {
                    name: gateType,
                    target: targetQubit,
                }
            ]
        };
        
        // Add control qubit if this is a controlled operation
        if (controlQubit !== null) {
            circuitDefinition.gates[0].control = controlQubit;
        }
        
        // Call the protected backend API
        const response = await fetch('/api/quantum/circuit', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                circuit: circuitDefinition,
                qubit_count: qubitCount
            })
        });
        
        // Check response status
        if (!response.ok) {
            const errorData = await response.json();
            console.error("Backend error:", errorData);
            throw new Error("Failed to process quantum operation on backend");
        }
        
        // Get results from backend
        const resultData = await response.json();
        
        if (resultData.success) {
            // If backend processing was successful, we can optionally update our frontend model
            // with the results from the proprietary engine
            console.log("Backend quantum operation successful", resultData);
        }
        
    } catch (error) {
        // Frontend continues to work even if backend call fails
        console.error("Error communicating with quantum backend:", error);
    }
}

// Reset quantum state to |0...0⟩
function resetQuantumState() {
    // Initialize using our EnhancedMultiQubitState class
    quantumState = new EnhancedMultiQubitState(qubitCount);
    currentState = Array(qubitCount).fill(0); // Still maintain simplified representation
    
    // Update visual elements
    updateQuantumState();
    updateBlochSphereVisualization();
    updateStatus('Quantum state reset to |0...0⟩');
}

// Update quantum state display
function updateQuantumState() {
    // Get probabilities from the quantum state
    const probabilities = quantumState.getProbabilities();
    
    // Find the most likely state for display
    let maxProb = 0;
    let maxState = '';
    for (const prob of probabilities) {
        if (prob.probability > maxProb) {
            maxProb = prob.probability;
            maxState = prob.state;
        }
    }
    
    // If no clear state, use our simplified representation
    if (maxState === '') {
        // Convert numerical state to ket notation
        let ketState = '|';
        for (let i = 0; i < currentState.length; i++) {
            if (currentState[i] === 0.5 || currentState[i] === -0.5) {
                // Superposition state
                ketState += '+'; // Simplified, should be + or - depending on sign
            } else {
                ketState += currentState[i];
            }
        }
        ketState += '⟩';
        elements.stateVector.textContent = ketState;
    } else {
        // Use the state from our quantum simulation
        elements.stateVector.textContent = `|${maxState}⟩`;
    }
    
    // Update probability display with real probabilities
    updateProbabilityDisplay(probabilities);
}

// Update probability bars
function updateProbabilityDisplay(probabilities) {
    // Clear current probability display
    elements.stateProbability.innerHTML = '';
    
    // Use probabilities from the quantum state if provided
    let states = [];
    
    if (probabilities) {
        // Filter to show only states with significant probability
        states = probabilities
            .filter(state => state.probability > 0.01) // Only show states with >1% probability
            .sort((a, b) => b.probability - a.probability) // Sort by descending probability
            .slice(0, 8); // Show at most 8 states to avoid cluttering
    } else {
        // Fallback for simplified model
        if (currentState.every(q => q === 0 || q === 1)) {
            // Basis state
            const basisState = currentState.join('');
            states.push({
                state: basisState,
                probability: 1.0
            });
        } else {
            // Superposition (simplified)
            states.push({
                state: '..', // Represents some superposition
                probability: 0.5,
                superposition: true
            });
            states.push({
                state: '..', // Represents some superposition
                probability: 0.5,
                superposition: true
            });
        }
    }
    
    // Create probability bars
    states.forEach(state => {
        const barContainer = document.createElement('div');
        barContainer.className = 'probability-bar';
        
        const barFill = document.createElement('div');
        barFill.className = 'bar-fill';
        barFill.style.width = `${state.probability * 100}%`;
        
        const barLabel = document.createElement('div');
        barLabel.className = 'bar-label';
        
        // Show state and probability percentage
        const stateText = state.state || '..';
        const probabilityText = `${Math.round(state.probability * 100)}%`;
        barLabel.textContent = `|${stateText}⟩: ${probabilityText}`;
        
        // Special styling for superposition
        if (state.superposition) {
            barFill.style.background = 'linear-gradient(90deg, #03a9f4, #9c27b0)';
            barLabel.textContent = 'Superposition state';
        }
        
        // Color gradient based on probability (0.0-1.0)
        // Higher probabilities are more vibrant
        if (!state.superposition) {
            const hue = 240 - state.probability * 200; // 240 (blue) to 40 (orange)
            const saturation = 60 + state.probability * 40; // 60-100%
            const lightness = 55 + state.probability * 10; // 55-65%
            barFill.style.background = `hsl(${hue}, ${saturation}%, ${lightness}%)`;
        }
        
        barContainer.appendChild(barFill);
        barContainer.appendChild(barLabel);
        elements.stateProbability.appendChild(barContainer);
    });
}

// Update algorithm description
function updateAlgorithmDescription() {
    const algorithmKey = elements.quantumAlgorithm.value;
    const algorithm = ALGORITHMS[algorithmKey];
    
    elements.algorithmDescription.innerHTML = `
        <h3>${algorithm.name}</h3>
        <p>${algorithm.description}</p>
    `;
    
    // Reset steps UI
    resetAlgorithmStepsUI();
    
    // Update step descriptions based on the selected algorithm
    const steps = document.querySelectorAll('.step');
    const stepCount = Math.min(steps.length, algorithm.steps.length);
    
    for (let i = 0; i < stepCount; i++) {
        const step = steps[i];
        const stepData = algorithm.steps[i];
        const stepDesc = step.querySelector('.step-desc');
        
        step.style.display = 'flex'; // Show all steps up to the algorithm's step count
        stepDesc.textContent = stepData.name;
    }
    
    // Hide any extra steps
    for (let i = stepCount; i < steps.length; i++) {
        steps[i].style.display = 'none';
    }
}

// Run the full algorithm
function runAlgorithm() {
    if (isAnimating) return;
    
    const algorithmKey = elements.quantumAlgorithm.value;
    currentAlgorithm = ALGORITHMS[algorithmKey];
    
    resetAlgorithm();
    
    // Start the algorithm animation
    isAnimating = true;
    runAlgorithmStep(0);
}

// Run algorithm step by step
function runAlgorithmStep(stepIndex) {
    if (stepIndex >= currentAlgorithm.steps.length) {
        isAnimating = false;
        updateStatus(`Algorithm complete: ${currentAlgorithm.name}`);
        return;
    }
    
    const step = currentAlgorithm.steps[stepIndex];
    algorithmStep = stepIndex + 1; // Steps are 1-indexed in UI
    
    // Update steps UI
    updateAlgorithmStepUI(algorithmStep);
    
    // Apply the gate for this step
    if (step.gate) {
        if (step.gate === 'custom') {
            // Custom operations would call into your Quantonium engine
            // For demonstration, we'll simulate some basic behaviors
            simulateCustomStep(step, stepIndex);
        } else {
            // Regular gate
            const target = step.target === 'all' ? 
                Array.from({length: qubitCount}, (_, i) => i) : 
                [step.target];
                
            target.forEach(t => {
                applyGateToState(step.gate, t, step.control);
            });
        }
    }
    
    updateQuantumState();
    updateStatus(`Step ${algorithmStep}: ${step.description}`);
    
    // Continue to next step
    const stepDelay = 1000 / animationSpeed;
    setTimeout(() => {
        runAlgorithmStep(stepIndex + 1);
    }, stepDelay);
}

// Step through algorithm one step at a time
function stepAlgorithm() {
    if (isAnimating) return;
    
    const algorithmKey = elements.quantumAlgorithm.value;
    
    if (!currentAlgorithm) {
        currentAlgorithm = ALGORITHMS[algorithmKey];
        resetAlgorithm();
    }
    
    if (algorithmStep >= currentAlgorithm.steps.length) {
        updateStatus('Algorithm complete. Reset to start again.');
        return;
    }
    
    // Run just one step
    isAnimating = true;
    runAlgorithmStep(algorithmStep - 1); // Convert from 1-indexed UI to 0-indexed array
    isAnimating = false;
}

// Reset algorithm state
function resetAlgorithm() {
    algorithmStep = 1;
    resetQuantumState();
    resetAlgorithmStepsUI();
    
    const algorithmKey = elements.quantumAlgorithm.value;
    currentAlgorithm = ALGORITHMS[algorithmKey];
    
    updateStatus(`Reset algorithm: ${currentAlgorithm.name}`);
}

// Update the UI for algorithm steps
function updateAlgorithmStepUI(stepNumber) {
    const steps = document.querySelectorAll('.step');
    
    steps.forEach(step => {
        const stepNum = parseInt(step.dataset.step);
        
        step.classList.remove('current', 'completed');
        
        if (stepNum < stepNumber) {
            step.classList.add('completed');
        } else if (stepNum === stepNumber) {
            step.classList.add('current');
        }
    });
}

// Reset the algorithm steps UI
function resetAlgorithmStepsUI() {
    const steps = document.querySelectorAll('.step');
    
    steps.forEach(step => {
        step.classList.remove('current', 'completed');
        if (parseInt(step.dataset.step) === 1) {
            step.classList.add('current');
        }
    });
}

// Simulate custom operations for algorithm steps
function simulateCustomStep(step, stepIndex) {
    const algorithmKey = elements.quantumAlgorithm.value;
    
    switch(algorithmKey) {
        case 'bell':
            // Bell state steps are handled by regular gates
            break;
            
        case 'grover':
            if (step.name === 'Oracle') {
                // Mark the |11⟩ state (simplified)
                updateStatus('Oracle applied: Marked |11⟩ state');
            } else if (step.name === 'Diffusion') {
                // Amplify marked state (simplified)
                updateStatus('Diffusion applied: Amplified marked state');
                // Set qubits to 1 to represent finding the solution
                currentState = Array(qubitCount).fill(1);
            }
            break;
            
        case 'deutsch':
            if (step.name === 'Initialize') {
                // Set second qubit to |1⟩
                if (qubitCount > 1) {
                    currentState[1] = 1;
                }
            } else if (step.name === 'Oracle') {
                // Simulate balanced function
                updateStatus('Oracle applied: Balanced function');
            }
            break;
            
        case 'teleport':
            if (step.name === 'Prepare') {
                // Prepare a state to teleport (e.g., |1⟩)
                currentState[0] = 1;
                updateStatus('Prepared state |1⟩ for teleportation');
            } else if (step.name === 'Bell Pair') {
                // Create Bell pair between qubits 1 and 2
                applyGateToState('h', 1);
                applyGateToState('cnot', 2, 1);
                updateStatus('Created Bell pair between qubits 1 and 2');
            } else if (step.name === 'Measure') {
                // Simulate measurement
                updateStatus('Measured qubits 0 and 1');
            } else if (step.name === 'Correction') {
                // Apply correction to qubit 2
                // This would teleport the state of qubit 0 to qubit 2
                currentState[2] = currentState[0];
                updateStatus('Applied correction gates to qubit 2');
            }
            break;
    }
}

// Update status message
function updateStatus(message) {
    elements.statusMessage.textContent = message;
}

// Initialize 3D visualization
function initVisualization() {
    // Set up Three.js scene
    scene = new THREE.Scene();
    scene.background = new THREE.Color(0x262626);
    
    // Create camera
    camera = new THREE.PerspectiveCamera(75, 
        elements.blochSphereContainer.clientWidth / elements.blochSphereContainer.clientHeight, 
        0.1, 1000);
    camera.position.z = 5;
    
    // Create renderer
    renderer = new THREE.WebGLRenderer({ antialias: true });
    renderer.setSize(elements.blochSphereContainer.clientWidth, elements.blochSphereContainer.clientHeight);
    elements.blochSphereContainer.appendChild(renderer.domElement);
    
    // Add orbit controls
    controls = new THREE.OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.dampingFactor = 0.25;
    
    // Create lighting
    const ambientLight = new THREE.AmbientLight(0x404040);
    scene.add(ambientLight);
    
    const directionalLight = new THREE.DirectionalLight(0xffffff, 0.5);
    directionalLight.position.set(1, 1, 1);
    scene.add(directionalLight);
    
    // Start animation loop
    animate();
    
    // Handle window resize
    window.addEventListener('resize', () => {
        camera.aspect = elements.blochSphereContainer.clientWidth / elements.blochSphereContainer.clientHeight;
        camera.updateProjectionMatrix();
        renderer.setSize(elements.blochSphereContainer.clientWidth, elements.blochSphereContainer.clientHeight);
    });
}

// Create Bloch sphere visualization
function createBlochSphere(index) {
    // Create sphere
    const geometry = new THREE.SphereGeometry(1, 32, 32);
    const material = new THREE.MeshPhongMaterial({ 
        color: 0x03a9f4,
        transparent: true,
        opacity: 0.6,
        wireframe: false
    });
    const sphere = new THREE.Mesh(geometry, material);
    
    // Position based on qubit index
    sphere.position.x = (index - (qubitCount - 1) / 2) * 3;
    
    // Add coordinate axes
    const axesHelper = new THREE.AxesHelper(1.2);
    sphere.add(axesHelper);
    
    // Add axis labels
    const createLabel = (text, position) => {
        const canvas = document.createElement('canvas');
        canvas.width = 64;
        canvas.height = 64;
        const ctx = canvas.getContext('2d');
        ctx.fillStyle = 'white';
        ctx.font = '48px Arial';
        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';
        ctx.fillText(text, 32, 32);
        
        const texture = new THREE.CanvasTexture(canvas);
        const material = new THREE.SpriteMaterial({ map: texture });
        const sprite = new THREE.Sprite(material);
        sprite.position.copy(position);
        sprite.scale.set(0.5, 0.5, 0.5);
        return sprite;
    };
    
    // Add |0⟩ and |1⟩ labels
    const zeroLabel = createLabel('|0⟩', new THREE.Vector3(0, 1.5, 0));
    const oneLabel = createLabel('|1⟩', new THREE.Vector3(0, -1.5, 0));
    sphere.add(zeroLabel);
    sphere.add(oneLabel);
    
    // Add qubit state vector
    const arrowDir = new THREE.Vector3(0, 1, 0);
    const arrowOrigin = new THREE.Vector3(0, 0, 0);
    const arrowLength = 1;
    const arrowColor = 0xffff00;
    const arrowHelper = new THREE.ArrowHelper(arrowDir, arrowOrigin, arrowLength, arrowColor, 0.1, 0.08);
    
    sphere.add(arrowHelper);
    scene.add(sphere);
    
    // Add label for qubit index
    const qubitLabel = createLabel(`Q${index}`, new THREE.Vector3(0, -2, 0));
    sphere.add(qubitLabel);
    
    return { sphere, arrow: arrowHelper };
}

// Update Bloch sphere visualization based on quantum state
function updateBlochSphereVisualization() {
    // Clear existing visualization
    while(scene.children.length > 0) { 
        scene.remove(scene.children[0]); 
    }
    
    qubitSpheres = [];
    qubitVectors = [];
    
    // Create a Bloch sphere for each qubit
    for (let i = 0; i < qubitCount; i++) {
        const { sphere, arrow } = createBlochSphere(i);
        qubitSpheres.push(sphere);
        qubitVectors.push(arrow);
        
        // Update arrow direction based on qubit state
        updateQubitVector(i);
    }
}

// Update qubit vector visualization
function updateQubitVector(index) {
    const state = currentState[index];
    const arrow = qubitVectors[index];
    
    let theta = 0; // polar angle (0 = |0⟩, π = |1⟩)
    let phi = 0;   // azimuthal angle
    
    if (state === 0) {
        theta = 0; // |0⟩ points up
    } else if (state === 1) {
        theta = Math.PI; // |1⟩ points down
    } else if (state === 0.5) {
        // |+⟩ state = (|0⟩ + |1⟩)/√2
        theta = Math.PI / 2;
        phi = 0;
    } else if (state === -0.5) {
        // |-⟩ state = (|0⟩ - |1⟩)/√2
        theta = Math.PI / 2;
        phi = Math.PI;
    }
    
    // Convert spherical coordinates to cartesian
    const x = Math.sin(theta) * Math.cos(phi);
    const y = Math.cos(theta);
    const z = Math.sin(theta) * Math.sin(phi);
    
    // Update arrow direction
    const direction = new THREE.Vector3(x, y, z);
    arrow.setDirection(direction.normalize());
}

// Animation loop
function animate() {
    requestAnimationFrame(animate);
    
    controls.update();
    
    // Rotate each Bloch sphere slightly
    qubitSpheres.forEach(sphere => {
        sphere.rotation.y += 0.005;
    });
    
    renderer.render(scene, camera);
}

// Switch visualization mode
function switchVisualizationMode() {
    const mode = elements.viewMode.value;
    
    if (mode === 'bloch') {
        elements.blochSphereContainer.style.display = 'block';
        // Hide other visualization modes
    } else if (mode === 'circuit') {
        elements.blochSphereContainer.style.display = 'none';
        // Show circuit diagram
    } else if (mode === 'matrix') {
        elements.blochSphereContainer.style.display = 'none';
        // Show matrix view
    }
}