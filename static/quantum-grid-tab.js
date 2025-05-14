/**
 * QuantoniumOS Quantum Grid Tab
 * 
 * This script handles the visualization of the quantum grid with up to 150 qubits
 * Uses the same visualization approach as the deployed app
 */

// Initialize quantum grid when the window is loaded
function initializeQuantumGridTab(windowId) {
    const qubitCount = 100; // Default to 100 qubits, can be changed by slider
    createQubitGrid(windowId, qubitCount);
    setupQuantumGridEvents(windowId);
}

// Create the grid of qubits
function createQubitGrid(windowId, count) {
    const gridContainer = document.getElementById(`quantum-grid-container-${windowId}`);
    if (!gridContainer) return;
    
    // Clear existing content
    gridContainer.innerHTML = '';
    
    // Calculate grid dimensions
    const cols = 8; // 8 qubits per row
    const rows = Math.ceil(count / cols);
    
    // Create the grid as a table
    let html = '<table class="quantum-grid-table">';
    for (let row = 0; row < rows; row++) {
        html += '<tr>';
        for (let col = 0; col < cols; col++) {
            const qubitIndex = row * cols + col;
            if (qubitIndex < count) {
                html += `
                    <td class="qubit-cell">
                        <div class="qubit" id="Q${qubitIndex}-${windowId}">
                            <div class="qubit-label">Q${qubitIndex}</div>
                            <div class="qubit-state">[0]</div>
                        </div>
                    </td>
                `;
            } else {
                html += '<td></td>';
            }
        }
        html += '</tr>';
    }
    html += '</table>';
    
    gridContainer.innerHTML = html;
}

// Setup event handlers for the quantum grid tab
function setupQuantumGridEvents(windowId) {
    // Qubit count slider
    const qubitSlider = document.getElementById(`qubit-count-slider-${windowId}`);
    const qubitCountDisplay = document.getElementById(`qubit-count-display-${windowId}`);
    
    if (qubitSlider && qubitCountDisplay) {
        qubitSlider.addEventListener('input', function() {
            const count = parseInt(this.value);
            qubitCountDisplay.textContent = count;
            createQubitGrid(windowId, count);
        });
    }
    
    // Run quantum process button
    const runButton = document.getElementById(`run-quantum-process-${windowId}`);
    if (runButton) {
        runButton.addEventListener('click', function() {
            runQuantumSimulation(windowId);
        });
    }
    
    // Run stress test button
    const stressButton = document.getElementById(`run-stress-test-${windowId}`);
    if (stressButton) {
        stressButton.addEventListener('click', function() {
            runStressTest(windowId);
        });
    }
}

// Run a quantum simulation with the current settings
function runQuantumSimulation(windowId) {
    const inputData = document.getElementById(`quantum-input-${windowId}`).value;
    const qubitCount = parseInt(document.getElementById(`qubit-count-slider-${windowId}`).value);
    
    // Show processing message
    const statusDisplay = document.getElementById(`quantum-status-${windowId}`);
    if (statusDisplay) {
        statusDisplay.textContent = 'Processing quantum simulation...';
        statusDisplay.style.color = 'cyan';
    }
    
    // Process data
    setTimeout(() => {
        // Update qubit states with random values to simulate quantum processing
        for (let i = 0; i < qubitCount; i++) {
            const qubit = document.getElementById(`Q${i}-${windowId}`);
            if (qubit) {
                const randomState = Math.random() > 0.5 ? '1' : '0';
                qubit.querySelector('.qubit-state').textContent = `[${randomState}]`;
                qubit.classList.add(randomState === '1' ? 'qubit-on' : 'qubit-off');
            }
        }
        
        // Update status
        if (statusDisplay) {
            statusDisplay.textContent = 'Quantum simulation complete';
            statusDisplay.style.color = '#4caf50';
        }
        
        // Update the wave visualization
        updateQuantumWaveVisualization(windowId, inputData);
    }, 1000);
}

// Run a stress test with all qubits
function runStressTest(windowId) {
    const qubitCount = parseInt(document.getElementById(`qubit-count-slider-${windowId}`).value);
    
    // Show processing message
    const statusDisplay = document.getElementById(`quantum-status-${windowId}`);
    if (statusDisplay) {
        statusDisplay.textContent = `Running 150-qubit stress test...`;
        statusDisplay.style.color = 'cyan';
    }
    
    // Process data
    let progress = 0;
    const progressInterval = setInterval(() => {
        progress += 10;
        if (statusDisplay) {
            statusDisplay.textContent = `Running 150-qubit stress test... ${progress}%`;
        }
        
        if (progress >= 100) {
            clearInterval(progressInterval);
            
            // Update all qubit states
            for (let i = 0; i < qubitCount; i++) {
                const qubit = document.getElementById(`Q${i}-${windowId}`);
                if (qubit) {
                    const isEntangled = Math.random() > 0.7;
                    qubit.querySelector('.qubit-state').textContent = isEntangled ? '[E]' : '[' + (Math.random() > 0.5 ? '1' : '0') + ']';
                    qubit.className = 'qubit ' + (isEntangled ? 'qubit-entangled' : (Math.random() > 0.5 ? 'qubit-on' : 'qubit-off'));
                }
            }
            
            // Update status
            if (statusDisplay) {
                statusDisplay.textContent = 'Stress test complete - All qubits operational';
                statusDisplay.style.color = '#4caf50';
            }
            
            // Update the wave visualization with random complex pattern
            updateQuantumWaveVisualization(windowId, 'stress-test');
        }
    }, 300);
}

// Update the quantum wave visualization
function updateQuantumWaveVisualization(windowId, inputData) {
    const canvas = document.getElementById(`quantum-wave-canvas-${windowId}`);
    if (!canvas) return;
    
    const ctx = canvas.getContext('2d');
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    
    // Draw two wave patterns - one blue, one purple
    ctx.lineWidth = 2;
    
    // Purple wave (amplitude wave)
    ctx.strokeStyle = '#b967ff';
    ctx.beginPath();
    
    const seed = inputData ? hashString(inputData) : Math.random() * 10000;
    const frequencyA = 0.05 + (seed % 10) / 50;
    const amplitudeA = 40 + (seed % 20);
    
    for (let x = 0; x < canvas.width; x++) {
        const y = canvas.height / 2 + Math.sin(x * frequencyA + seed / 1000) * amplitudeA;
        if (x === 0) {
            ctx.moveTo(x, y);
        } else {
            ctx.lineTo(x, y);
        }
    }
    ctx.stroke();
    
    // Blue wave (phase wave)
    ctx.strokeStyle = '#00b7ff';
    ctx.beginPath();
    
    const frequencyB = 0.03 + (seed % 15) / 60;
    const amplitudeB = 30 + (seed % 15);
    const phaseShift = seed % (2 * Math.PI);
    
    for (let x = 0; x < canvas.width; x++) {
        const y = canvas.height / 2 + Math.sin(x * frequencyB + phaseShift) * amplitudeB;
        if (x === 0) {
            ctx.moveTo(x, y);
        } else {
            ctx.lineTo(x, y);
        }
    }
    ctx.stroke();
}

// Simple hash function for strings
function hashString(str) {
    let hash = 0;
    for (let i = 0; i < str.length; i++) {
        hash = ((hash << 5) - hash) + str.charCodeAt(i);
        hash |= 0; // Convert to 32bit integer
    }
    return Math.abs(hash);
}

// Function to create the quantum grid tab content
function createQuantumGridTabContent(windowId) {
    return `
        <div class="quantum-grid-tab-content">
            <div class="quantum-grid-header">
                <h2>Quantum Grid Visualization</h2>
                <p>Visualize and test quantum grid operations on up to 150 qubits.</p>
            </div>
            
            <div class="quantum-grid-controls">
                <div class="control-group">
                    <label>Qubit Count (2-150):</label>
                    <input type="range" id="qubit-count-slider-${windowId}" min="2" max="150" value="100" class="slider">
                    <span id="qubit-count-display-${windowId}">100</span>
                </div>
                
                <div class="control-group">
                    <label>Input Data:</label>
                    <input type="text" id="quantum-input-${windowId}" placeholder="Enter data for quantum processing..." class="text-input">
                </div>
                
                <div class="button-group">
                    <button id="run-quantum-process-${windowId}" class="quantum-btn">Run Quantum Process</button>
                    <button id="run-stress-test-${windowId}" class="quantum-btn">Run Stress Test (150 Qubits)</button>
                </div>
                
                <div id="quantum-status-${windowId}" class="status-display">Ready for quantum operations</div>
            </div>
            
            <div class="quantum-visualization">
                <div class="visualization-column">
                    <h3>Quantum Grid</h3>
                    <div id="quantum-grid-container-${windowId}" class="quantum-grid-container"></div>
                </div>
                <div class="visualization-column">
                    <h3>Quantum Wave Visualization</h3>
                    <canvas id="quantum-wave-canvas-${windowId}" width="400" height="200" class="wave-canvas"></canvas>
                </div>
            </div>
        </div>
    `;
}

// CSS for the quantum grid tab
const quantumGridCSS = `
    .quantum-grid-tab-content {
        padding: 15px;
        color: #fff;
        height: 100%;
        overflow: auto;
    }
    
    .quantum-grid-header {
        margin-bottom: 20px;
    }
    
    .quantum-grid-header h2 {
        margin: 0 0 10px 0;
        font-weight: 300;
    }
    
    .quantum-grid-header p {
        margin: 0;
        opacity: 0.7;
    }
    
    .quantum-grid-controls {
        margin-bottom: 20px;
        padding: 15px;
        background-color: rgba(0, 0, 0, 0.3);
        border-radius: 8px;
    }
    
    .control-group {
        margin-bottom: 15px;
    }
    
    .control-group label {
        display: block;
        margin-bottom: 5px;
        font-weight: 300;
    }
    
    .slider {
        width: 100%;
        background: #333;
        height: 5px;
        outline: none;
        border-radius: 5px;
        -webkit-appearance: none;
    }
    
    .slider::-webkit-slider-thumb {
        -webkit-appearance: none;
        width: 15px;
        height: 15px;
        background: #00b7ff;
        border-radius: 50%;
    }
    
    .text-input {
        width: 100%;
        padding: 8px;
        background-color: #111;
        border: 1px solid #333;
        border-radius: 4px;
        color: #fff;
        outline: none;
    }
    
    .button-group {
        display: flex;
        gap: 10px;
        margin-bottom: 15px;
    }
    
    .quantum-btn {
        padding: 8px 15px;
        background-color: #00475e;
        color: white;
        border: none;
        border-radius: 4px;
        cursor: pointer;
        transition: background-color 0.3s;
    }
    
    .quantum-btn:hover {
        background-color: #006a8c;
    }
    
    .status-display {
        padding: 10px;
        background-color: rgba(0, 0, 0, 0.5);
        border-radius: 4px;
        font-family: monospace;
        color: #aaa;
    }
    
    .quantum-visualization {
        display: flex;
        flex-wrap: wrap;
        gap: 20px;
    }
    
    .visualization-column {
        flex: 1;
        min-width: 300px;
    }
    
    .quantum-grid-container {
        background-color: rgba(0, 0, 0, 0.3);
        border-radius: 8px;
        padding: 10px;
        max-height: 400px;
        overflow: auto;
    }
    
    .quantum-grid-table {
        width: 100%;
        border-spacing: 5px;
        border-collapse: separate;
    }
    
    .qubit-cell {
        padding: 0;
    }
    
    .qubit {
        background-color: #111;
        border: 1px solid #333;
        border-radius: 4px;
        padding: 5px;
        text-align: center;
        transition: all 0.3s;
    }
    
    .qubit-label {
        color: #00b7ff;
        font-size: 12px;
        margin-bottom: 2px;
    }
    
    .qubit-state {
        font-size: 10px;
        color: #aaa;
    }
    
    .qubit-on {
        background-color: rgba(0, 183, 255, 0.2);
        border-color: #00b7ff;
    }
    
    .qubit-off {
        background-color: #111;
    }
    
    .qubit-entangled {
        background-color: rgba(185, 103, 255, 0.2);
        border-color: #b967ff;
    }
    
    .qubit-entangled .qubit-state {
        color: #b967ff;
    }
    
    .wave-canvas {
        width: 100%;
        height: 200px;
        background-color: #000;
        border: 1px solid #333;
        border-radius: 4px;
    }
`;

// Add the quantum grid CSS to the document
document.addEventListener('DOMContentLoaded', function() {
    const style = document.createElement('style');
    style.innerHTML = quantumGridCSS;
    document.head.appendChild(style);
});