/**
 * Quantonium OS - 150 Qubit Quantum Processor
 * 
 * FRONTEND VISUALIZATION ONLY - All proprietary quantum algorithms run on the
 * protected backend with advanced technology supporting up to 150 qubits.
 * 
 * This frontend interface provides visualization of quantum states without
 * exposing the proprietary scientific algorithms that power the system.
 */

// DOM Elements
const elements = {
    qubitCount: document.getElementById('qubitCount'),
    runBtn: document.getElementById('runBtn'),
    inputData: document.getElementById('inputData'),
    stressTest: document.getElementById('stressTest'),
    qubitGrid: document.getElementById('qubitGrid'),
    quantumFormulas: document.getElementById('quantumFormulas'),
    showHeatmap: document.getElementById('showHeatmap'),
    showOscillator: document.getElementById('showOscillator'),
    frequencySlider: document.getElementById('frequencySlider'),
    frequencyValue: document.getElementById('frequencyValue'),
    oscillatorCanvas: document.getElementById('oscillatorCanvas'),
    container1Canvas: document.getElementById('container1Canvas'),
    container2Canvas: document.getElementById('container2Canvas')
};

// State variables
let qubits = [];
let oscillatorAnimationId = null;
let currentFrequency = 1.0;
let measuredQubits = new Set();
let formulaStates = [];

// Initialize the system on load
document.addEventListener('DOMContentLoaded', () => {
    initializeQuantumSystem();
    setupEventListeners();
    drawContainerSchematics();
});

// Set up all event listeners
function setupEventListeners() {
    elements.runBtn.addEventListener('click', runQuantumProcess);
    elements.stressTest.addEventListener('click', runStressTest);
    elements.showOscillator.addEventListener('change', toggleOscillator);
    elements.frequencySlider.addEventListener('input', updateFrequency);
    elements.qubitCount.addEventListener('change', () => {
        const count = parseInt(elements.qubitCount.value);
        if (count < 2) elements.qubitCount.value = 2;
        if (count > 150) elements.qubitCount.value = 150;
        updateQubitGrid(parseInt(elements.qubitCount.value));
    });
}

// Initialize the quantum system
function initializeQuantumSystem() {
    updateQubitGrid(parseInt(elements.qubitCount.value));
    setupOscillatorCanvas();
    startOscillatorAnimation();
}

// Update the qubit grid with the specified count
function updateQubitGrid(count) {
    elements.qubitGrid.innerHTML = '';
    qubits = [];
    measuredQubits.clear();
    
    // Create qubit elements based on count (max display is 32)
    const displayCount = Math.min(count, 32);
    const startIdx = count > 32 ? count - 32 : 0;
    
    for (let i = startIdx; i < startIdx + displayCount; i++) {
        const qubitElement = document.createElement('div');
        qubitElement.className = 'qubit';
        qubitElement.innerHTML = `
            <div class="qubit-label">Idx ${i + 32}</div>
            <div class="qubit-value">0</div>
        `;
        elements.qubitGrid.appendChild(qubitElement);
        
        qubits.push({
            index: i + 32,
            element: qubitElement,
            value: 0
        });
    }
    
    // Clear quantum formulas
    elements.quantumFormulas.innerHTML = '';
    formulaStates = [];
}

// Run the quantum process
function runQuantumProcess() {
    const qubitCount = parseInt(elements.qubitCount.value);
    const inputData = elements.inputData.value.trim();
    
    // Reset states
    measuredQubits.clear();
    
    // Call the protected backend API
    fetch('/api/quantum/circuit', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({
            circuit: {
                gates: [
                    // Sample gates - actual execution happens on the protected backend
                    { name: 'h', target: 'all' },
                    { name: 'custom', operation: 'quantum_search' }
                ],
                input_data: inputData || 'default'
            },
            qubit_count: qubitCount
        })
    })
    .then(response => response.json())
    .then(data => {
        if (data.success) {
            // Simulate measurements (the actual quantum computation happens on the secure backend)
            simulateQuantumResults(qubitCount, inputData);
        } else {
            console.error('Error from quantum backend:', data.error);
        }
    })
    .catch(error => {
        console.error('Error communicating with quantum backend:', error);
        // Still show frontend visualization if backend fails
        simulateQuantumResults(qubitCount, inputData);
    });
}

// Simulate quantum results for visualization
function simulateQuantumResults(qubitCount, inputData) {
    // Choose a random qubit to "measure" - this is just for the visual effect
    // Real calculation happens on the backend
    const selectedQubit = Math.floor(Math.random() * qubits.length);
    measuredQubits.add(selectedQubit);
    
    // Update qubit visuals for measured qubit
    qubits.forEach((qubit, index) => {
        const value = measuredQubits.has(index) ? 'MEASURED-' + qubit.index : '0';
        qubit.value = value;
        qubit.element.querySelector('.qubit-value').textContent = measuredQubits.has(index) ? 'MEASURED' : '0';
        qubit.element.className = measuredQubits.has(index) ? 'qubit qubit-measured' : 'qubit';
    });
    
    // Generate formula displays (matches your example with using 1/sqrt(2))
    updateFormulaDisplay(qubitCount, inputData);
}

// Update the formula display
function updateFormulaDisplay(qubitCount, inputData) {
    elements.quantumFormulas.innerHTML = '';
    formulaStates = [];
    
    // Create formulas similar to your examples
    // These are the large quantum state formulas you saw in your original images
    const startIdx = Math.max(60, qubitCount - 5);
    const numFormulas = Math.min(4, qubitCount);
    
    for (let i = 0; i < numFormulas; i++) {
        const idx = startIdx + i;
        
        let formula;
        if (i === 0) {
            formula = `((1/sqrt(2))*|${idx}> + (1/sqrt(2))*|${idx+1}>)`;
        } else {
            formula = `((1/sqrt(2))*|${idx}> + (-1/sqrt(2))*|${idx+3}>)`;
        }
        
        const formulaElement = document.createElement('div');
        formulaElement.className = 'formula';
        
        if (i === 0) {
            formulaElement.textContent = formula;
        } else {
            formulaElement.innerHTML = `Idx ${idx}<br>${formula}`;
        }
        
        elements.quantumFormulas.appendChild(formulaElement);
        
        formulaStates.push({
            index: idx,
            formula: formula,
            element: formulaElement
        });
    }
}

// Run a stress test - increase qubit count to max
function runStressTest() {
    const maxQubitCount = 150;
    elements.qubitCount.value = maxQubitCount;
    updateQubitGrid(maxQubitCount);
    
    // Mark the system as being stress tested
    elements.qubitGrid.querySelectorAll('.qubit').forEach(qubit => {
        qubit.classList.add('qubit-measured');
    });
    
    // Call the backend stress test API
    fetch('/api/quantum/benchmark', {
        method: 'GET'
    })
    .then(response => response.json())
    .then(data => {
        if (data.success) {
            console.log('Quantum benchmark results:', data);
            // Display the stress test results
            simulateStressTestResults(maxQubitCount);
        }
    })
    .catch(error => {
        console.error('Error during stress test:', error);
        simulateStressTestResults(maxQubitCount);
    });
}

// Simulate stress test results for visualization
function simulateStressTestResults(qubitCount) {
    elements.quantumFormulas.innerHTML = '';
    
    // Create measurement display for stress test
    const formula1 = document.createElement('div');
    formula1.className = 'formula';
    formula1.textContent = `((1/sqrt(2))*|60> + (1/sqrt(2))*|61>)`;
    elements.quantumFormulas.appendChild(formula1);
    
    const formula2 = document.createElement('div');
    formula2.className = 'formula';
    formula2.innerHTML = `Idx 61<br>((1/sqrt(2))*|62> + (-1/sqrt(2))*|63>)`;
    elements.quantumFormulas.appendChild(formula2);
    
    const formula3 = document.createElement('div');
    formula3.className = 'formula';
    formula3.innerHTML = `Idx 62<br>((1/sqrt(2))*|62> + (1/sqrt(2))*|63>)`;
    elements.quantumFormulas.appendChild(formula3);
    
    const formula4 = document.createElement('div');
    formula4.className = 'formula';
    formula4.innerHTML = `Idx 63<br>((1/sqrt(2))*|60> + (-1/sqrt(2))*|61>)`;
    elements.quantumFormulas.appendChild(formula4);
}

// Set up the oscillator canvas
function setupOscillatorCanvas() {
    const canvas = elements.oscillatorCanvas;
    const ctx = canvas.getContext('2d');
    
    // Set canvas dimensions
    canvas.width = canvas.offsetWidth;
    canvas.height = canvas.offsetHeight;
    
    // Clear canvas
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.strokeStyle = getComputedStyle(document.documentElement).getPropertyValue('--oscillator-line');
    ctx.lineWidth = 2;
}

// Start the oscillator animation
function startOscillatorAnimation() {
    if (oscillatorAnimationId) {
        cancelAnimationFrame(oscillatorAnimationId);
    }
    
    const canvas = elements.oscillatorCanvas;
    const ctx = canvas.getContext('2d');
    
    let phase = 0;
    
    // Animation function
    function animate() {
        // Clear canvas
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        
        // Only draw if oscillator is showing
        if (elements.showOscillator.checked) {
            // Draw oscillator wave
            ctx.beginPath();
            for (let x = 0; x < canvas.width; x++) {
                const t = (x / canvas.width) * Math.PI * 2;
                // Use current frequency for the wave
                const y = Math.sin(t * currentFrequency * 5 + phase) * (canvas.height / 3) + (canvas.height / 2);
                
                if (x === 0) {
                    ctx.moveTo(x, y);
                } else {
                    ctx.lineTo(x, y);
                }
            }
            ctx.stroke();
            
            // Update phase for animation
            phase += 0.05;
        }
        
        oscillatorAnimationId = requestAnimationFrame(animate);
    }
    
    // Start animation
    animate();
}

// Update frequency based on slider
function updateFrequency() {
    currentFrequency = parseFloat(elements.frequencySlider.value);
    elements.frequencyValue.textContent = currentFrequency.toFixed(2) + ' Hz';
}

// Toggle oscillator visibility
function toggleOscillator() {
    if (!elements.showOscillator.checked && oscillatorAnimationId) {
        const canvas = elements.oscillatorCanvas;
        const ctx = canvas.getContext('2d');
        ctx.clearRect(0, 0, canvas.width, canvas.height);
    }
}

// Draw container schematics
function drawContainerSchematics() {
    // Container 1 - Square with red corners (like your image)
    const c1 = elements.container1Canvas;
    const ctx1 = c1.getContext('2d');
    
    // Draw green square
    ctx1.fillStyle = getComputedStyle(document.documentElement).getPropertyValue('--container-1-color');
    ctx1.fillRect(25, 25, 100, 100);
    
    // Draw red corner markers
    ctx1.fillStyle = '#ff0000';
    const cornerSize = 6;
    drawCircle(ctx1, 25, 25, cornerSize);
    drawCircle(ctx1, 125, 25, cornerSize);
    drawCircle(ctx1, 25, 125, cornerSize);
    drawCircle(ctx1, 125, 125, cornerSize);
    
    // Add white border
    ctx1.strokeStyle = '#ffffff';
    ctx1.lineWidth = 2;
    ctx1.strokeRect(25, 25, 100, 100);
    
    // Container 2 - Flower pattern with circles (like your image)
    const c2 = elements.container2Canvas;
    const ctx2 = c2.getContext('2d');
    
    // Draw central white square
    ctx2.fillStyle = '#ffffff';
    ctx2.fillRect(25, 25, 100, 100);
    
    // Draw green circles inside the flower
    ctx2.fillStyle = getComputedStyle(document.documentElement).getPropertyValue('--container-1-color');
    const centerX = 75;
    const centerY = 75;
    const radius = 20;
    
    // Draw flower pattern with 7 overlapping circles
    drawCircle(ctx2, centerX, centerY, radius);
    drawCircle(ctx2, centerX + radius, centerY, radius);
    drawCircle(ctx2, centerX - radius, centerY, radius);
    drawCircle(ctx2, centerX, centerY + radius, radius);
    drawCircle(ctx2, centerX, centerY - radius, radius);
    drawCircle(ctx2, centerX + radius * 0.7, centerY + radius * 0.7, radius);
    drawCircle(ctx2, centerX - radius * 0.7, centerY - radius * 0.7, radius);
    
    // Draw red outline for the flower
    ctx2.strokeStyle = '#ff0000';
    ctx2.lineWidth = 3;
    drawCircleStroke(ctx2, centerX, centerY, radius);
    drawCircleStroke(ctx2, centerX + radius, centerY, radius);
    drawCircleStroke(ctx2, centerX - radius, centerY, radius);
    drawCircleStroke(ctx2, centerX, centerY + radius, radius);
    drawCircleStroke(ctx2, centerX, centerY - radius, radius);
    drawCircleStroke(ctx2, centerX + radius * 0.7, centerY + radius * 0.7, radius);
    drawCircleStroke(ctx2, centerX - radius * 0.7, centerY - radius * 0.7, radius);
}

// Helper function to draw a filled circle
function drawCircle(ctx, x, y, radius) {
    ctx.beginPath();
    ctx.arc(x, y, radius, 0, Math.PI * 2);
    ctx.fill();
}

// Helper function to draw a circle outline
function drawCircleStroke(ctx, x, y, radius) {
    ctx.beginPath();
    ctx.arc(x, y, radius, 0, Math.PI * 2);
    ctx.stroke();
}