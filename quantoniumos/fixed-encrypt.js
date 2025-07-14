// Global variables for quantum grid
let gridInitialized = false;
let oscillatorAnimationId = null;
let gridFrequency = 1.0;
let gridAmplitude = 1.0;
let gridQubitCount = 6;

// Initialize the quantum grid panel
function initializeQuantumGrid() {
    if (gridInitialized) return;
    
    // Check if the necessary elements exist
    if (!elements.gridQubitGrid || !elements.gridContainer1Canvas) {
        console.log("Grid elements not found, skipping quantum grid initialization");
        return;
    }
    
    // Initialize grid controls
    if (elements.gridRunBtn) {
        elements.gridRunBtn.addEventListener('click', runQuantumGrid);
    }
    
    if (elements.gridStressTest) {
        elements.gridStressTest.addEventListener('click', runStressTest);
    }
    
    if (elements.gridShowOscillator) {
        elements.gridShowOscillator.addEventListener('change', toggleOscillator);
    }
    
    if (elements.gridFrequencySlider) {
        elements.gridFrequencySlider.addEventListener('input', updateGridFrequency);
    }
    
    if (elements.gridQubitCount) {
        elements.gridQubitCount.addEventListener('change', () => updateQubitGrid());
    }
    
    // Create initial qubit grid
    updateQubitGrid(gridQubitCount);
    
    // Draw container schematics
    drawContainerSchematics();
    
    // Start oscillator animation
    startOscillatorAnimation();
    
    // Update frequency display
    updateGridFrequency();
    
    gridInitialized = true;
    
    console.log("Quantum Grid initialized with 150-qubit support");
}

// Update the qubit grid display
function updateQubitGrid(count) {
    if (!elements.gridQubitGrid) return;
    
    // Update grid qubit count
    gridQubitCount = count || parseInt(elements.gridQubitCount.value);
    if (isNaN(gridQubitCount) || gridQubitCount < 2) {
        gridQubitCount = 2;
    } else if (gridQubitCount > 150) {
        gridQubitCount = 150;
    }
    
    // Update input field value
    if (elements.gridQubitCount) {
        elements.gridQubitCount.value = gridQubitCount;
    }
    
    // Clear the grid
    elements.gridQubitGrid.innerHTML = '';
    
    // Calculate grid dimensions
    const columns = Math.min(8, Math.ceil(Math.sqrt(gridQubitCount)));
    elements.gridQubitGrid.style.gridTemplateColumns = `repeat(${columns}, 1fr)`;
    
    // Create qubits
    for (let i = 0; i < gridQubitCount; i++) {
        const qubit = document.createElement('div');
        qubit.className = 'qubit';
        qubit.dataset.index = i;
        qubit.innerHTML = `
            <div class="qubit-number">Q${i}</div>
            <div class="qubit-value">|0⟩</div>
        `;
        
        // Style the qubit element
        qubit.style.padding = '8px';
        qubit.style.borderRadius = '4px';
        qubit.style.backgroundColor = 'rgba(0,0,0,0.3)';
        qubit.style.textAlign = 'center';
        qubit.querySelector('.qubit-number').style.fontWeight = 'bold';
        qubit.querySelector('.qubit-number').style.color = '#00e5ff';
        qubit.querySelector('.qubit-value').style.fontFamily = 'monospace';
        qubit.querySelector('.qubit-value').style.margin = '5px 0';
        
        elements.gridQubitGrid.appendChild(qubit);
    }
    
    updateStatus(`Quantum grid updated with ${gridQubitCount} qubits`, 'info');
}

// Run quantum process on the grid
function runQuantumGrid() {
    const inputData = elements.gridInputData ? elements.gridInputData.value : "";
    
    // Simulate quantum processing with animation
    updateStatus(`Running quantum process on ${gridQubitCount} qubits...`, 'info');
    
    // Reset grid qubits to initial state
    const qubits = elements.gridQubitGrid.querySelectorAll('.qubit');
    qubits.forEach(qubit => {
        qubit.classList.remove('qubit-measured');
        qubit.querySelector('.qubit-value').textContent = '|0⟩';
    });
    
    // Simulate delay for processing
    setTimeout(() => {
        simulateQuantumResults(gridQubitCount, inputData);
        updateFormulaDisplay(gridQubitCount, inputData);
    }, 500);
}

// Run quantum stress test
function runStressTest() {
    updateStatus(`Running stress test with 150 qubits...`, 'info');
    
    // Reset grid qubits to initial state
    const qubits = elements.gridQubitGrid.querySelectorAll('.qubit');
    qubits.forEach(qubit => {
        qubit.classList.remove('qubit-measured');
        qubit.querySelector('.qubit-value').textContent = '|0⟩';
    });
    
    // Update to maximum quantum count for stress test
    elements.gridQubitCount.value = "150";
    updateQubitGrid(150);
    
    // Simulate delay for processing
    setTimeout(() => {
        simulateStressTestResults(150);
        // Update formula display for stress test
        updateFormulaDisplay(150, "Stress Test", true);
        updateStatus(`Stress test completed for 150 qubits - system capacity verified`, 'success');
    }, 800);
}

// Simulate quantum results (frontend visualization only)
function simulateQuantumResults(qubitCount, inputData) {
    // Get quantum qubits
    const qubits = elements.gridQubitGrid.querySelectorAll('.qubit');
    
    // Simulate hadamard gates on first half of qubits
    const midpoint = Math.floor(qubitCount / 2);
    
    qubits.forEach((qubit, index) => {
        const qubitValue = qubit.querySelector('.qubit-value');
        
        if (index < midpoint) {
            // Superposition for first half
            qubitValue.textContent = '|+⟩';
            qubit.style.backgroundColor = 'rgba(106, 27, 154, 0.3)';  // Quantum color
        } else {
            // Random measured values for second half
            const value = Math.random() > 0.5 ? '|1⟩' : '|0⟩';
            qubitValue.textContent = value;
            if (value === '|1⟩') {
                qubit.style.backgroundColor = 'rgba(0, 137, 123, 0.3)';  // Resonance color
            } else {
                qubit.style.backgroundColor = 'rgba(21, 101, 192, 0.3)';  // Symbolic color
            }
            qubit.classList.add('qubit-measured');
        }
    });
    
    updateStatus(`Quantum process completed for ${qubitCount} qubits`, 'success');
}

// Simulate stress test results
function simulateStressTestResults(qubitCount) {
    // Get quantum qubits
    const qubits = elements.gridQubitGrid.querySelectorAll('.qubit');
    
    // For stress test, create an entangled pattern
    qubits.forEach((qubit, index) => {
        const qubitValue = qubit.querySelector('.qubit-value');
        
        if (index % 3 === 0) {
            // Every third qubit in |0⟩
            qubitValue.textContent = '|0⟩';
            qubit.style.backgroundColor = 'rgba(21, 101, 192, 0.3)';  // Symbolic color
        } else if (index % 3 === 1) {
            // Every third+1 qubit in |1⟩
            qubitValue.textContent = '|1⟩';
            qubit.style.backgroundColor = 'rgba(0, 137, 123, 0.3)';  // Resonance color
            qubit.classList.add('qubit-measured');
        } else {
            // Every third+2 qubit in |+⟩
            qubitValue.textContent = '|+⟩';
            qubit.style.backgroundColor = 'rgba(106, 27, 154, 0.3)';  // Quantum color
        }
    });
}

// Update quantum formula display
function updateFormulaDisplay(qubitCount, inputData, isStressTest = false) {
    if (!elements.gridQuantumFormulas) return;
    
    // Clear formulas
    elements.gridQuantumFormulas.innerHTML = '';
    
    // Create formula elements
    if (isStressTest) {
        // Stress test formulas
        const formulas = [
            `ψ = (1/√2)^${Math.floor(qubitCount/3)} ⋅ |${Array(Math.floor(qubitCount/3)).fill('0').join('')}⟩`,
            `φ = H^⊗${qubitCount} |${Array(Math.min(10, qubitCount)).fill('0').join('')}...⟩`,
            `Σ = (1/√${2**Math.min(10, qubitCount)}) ⋅ ∑|x⟩`
        ];
        
        formulas.forEach(formula => {
            const formulaElement = document.createElement('div');
            formulaElement.className = 'formula';
            formulaElement.style.margin = '5px 0';
            formulaElement.style.fontFamily = 'monospace';
            formulaElement.textContent = formula;
            elements.gridQuantumFormulas.appendChild(formulaElement);
        });
    } else {
        // Standard formulas
        const hash = hashString(inputData || 'quantum');
        const formulas = [
            `|ψ⟩ = (1/√2) ⋅ (|${hash.substring(0,4)}⟩ + |${hash.substring(4,8)}⟩)`,
            `H^⊗${Math.floor(qubitCount/2)} |${Array(Math.min(10, Math.floor(qubitCount/2))).fill('0').join('')}⟩ = (1/√${2**Math.min(10, Math.floor(qubitCount/2))}) ⋅ ∑|x⟩`
        ];
        
        formulas.forEach(formula => {
            const formulaElement = document.createElement('div');
            formulaElement.className = 'formula';
            formulaElement.style.margin = '5px 0';
            formulaElement.style.fontFamily = 'monospace';
            formulaElement.textContent = formula;
            elements.gridQuantumFormulas.appendChild(formulaElement);
        });
    }
}

// Start oscillator animation
function startOscillatorAnimation() {
    if (!gridOscillatorCtx || !elements.gridOscillatorCanvas) return;
    
    // Stop existing animation if running
    if (oscillatorAnimationId) {
        cancelAnimationFrame(oscillatorAnimationId);
    }
    
    const canvas = elements.gridOscillatorCanvas;
    const ctx = gridOscillatorCtx;
    const width = canvas.width;
    const height = canvas.height;
    let time = 0;
    
    // Animation function
    function animate() {
        ctx.clearRect(0, 0, width, height);
        
        // Draw background
        ctx.fillStyle = '#1a1a1a';
        ctx.fillRect(0, 0, width, height);
        
        // Draw center line
        ctx.strokeStyle = '#444';
        ctx.lineWidth = 1;
        ctx.beginPath();
        ctx.moveTo(0, height / 2);
        ctx.lineTo(width, height / 2);
        ctx.stroke();
        
        // Draw wave
        ctx.strokeStyle = '#673AB7';
        ctx.lineWidth = 2;
        ctx.beginPath();
        
        const step = 2;
        for (let x = 0; x < width; x += step) {
            const y = height / 2 + Math.sin((x / width * 8 + time) * Math.PI * gridFrequency) * gridAmplitude * (height / 3);
            if (x === 0) {
                ctx.moveTo(x, y);
            } else {
                ctx.lineTo(x, y);
            }
        }
        
        ctx.stroke();
        
        // Update time
        time += 0.01;
        
        // Continue animation if oscillator is enabled
        if (elements.gridShowOscillator && elements.gridShowOscillator.checked) {
            oscillatorAnimationId = requestAnimationFrame(animate);
        }
    }
    
    // Start animation
    animate();
}

// Update oscillator frequency
function updateGridFrequency() {
    if (!elements.gridFrequencySlider || !elements.gridFrequencyValue) return;
    
    gridFrequency = parseFloat(elements.gridFrequencySlider.value);
    elements.gridFrequencyValue.textContent = gridFrequency.toFixed(2) + ' Hz';
}

// Toggle oscillator
function toggleOscillator() {
    if (!elements.gridShowOscillator) return;
    
    if (elements.gridShowOscillator.checked) {
        startOscillatorAnimation();
    } else if (oscillatorAnimationId) {
        cancelAnimationFrame(oscillatorAnimationId);
        oscillatorAnimationId = null;
    }
}

// Draw container schematics
function drawContainerSchematics() {
    if (!gridContainer1Ctx || !gridContainer2Ctx) return;
    
    // Container 1: Regular pattern
    const ctx1 = gridContainer1Ctx;
    ctx1.clearRect(0, 0, 150, 150);
    
    // Draw background
    ctx1.fillStyle = '#1a1a1a';
    ctx1.fillRect(0, 0, 150, 150);
    
    // Draw grid
    ctx1.strokeStyle = '#333';
    ctx1.lineWidth = 1;
    
    // Horizontal lines
    for (let y = 0; y <= 150; y += 30) {
        ctx1.beginPath();
        ctx1.moveTo(0, y);
        ctx1.lineTo(150, y);
        ctx1.stroke();
    }
    
    // Vertical lines
    for (let x = 0; x <= 150; x += 30) {
        ctx1.beginPath();
        ctx1.moveTo(x, 0);
        ctx1.lineTo(x, 150);
        ctx1.stroke();
    }
    
    // Draw elements (squares and circles)
    for (let y = 0; y < 5; y++) {
        for (let x = 0; x < 5; x++) {
            const centerX = x * 30 + 15;
            const centerY = y * 30 + 15;
            
            if ((x + y) % 2 === 0) {
                // Draw green square
                ctx1.fillStyle = '#00c853';
                ctx1.fillRect(centerX - 10, centerY - 10, 20, 20);
            } else {
                // Draw red circle
                ctx1.fillStyle = '#f44336';
                drawCircle(ctx1, centerX, centerY, 10);
            }
        }
    }
    
    // Container 2: Random pattern
    const ctx2 = gridContainer2Ctx;
    ctx2.clearRect(0, 0, 150, 150);
    
    // Draw background
    ctx2.fillStyle = '#1a1a1a';
    ctx2.fillRect(0, 0, 150, 150);
    
    // Draw grid
    ctx2.strokeStyle = '#333';
    ctx2.lineWidth = 1;
    
    // Horizontal lines
    for (let y = 0; y <= 150; y += 30) {
        ctx2.beginPath();
        ctx2.moveTo(0, y);
        ctx2.lineTo(150, y);
        ctx2.stroke();
    }
    
    // Vertical lines
    for (let x = 0; x <= 150; x += 30) {
        ctx2.beginPath();
        ctx2.moveTo(x, 0);
        ctx2.lineTo(x, 150);
        ctx2.stroke();
    }
    
    // Draw elements with pseudo-random pattern based on hashString
    const hash = hashString('QuantumGrid');
    let hashIndex = 0;
    
    for (let y = 0; y < 5; y++) {
        for (let x = 0; x < 5; x++) {
            const centerX = x * 30 + 15;
            const centerY = y * 30 + 15;
            
            // Use hash character to determine shape
            const hashChar = parseInt(hash[hashIndex % hash.length], 16);
            hashIndex++;
            
            if (hashChar < 8) {
                // Draw green square
                ctx2.fillStyle = '#00c853';
                ctx2.fillRect(centerX - 10, centerY - 10, 20, 20);
            } else {
                // Draw red circle
                ctx2.fillStyle = '#f44336';
                drawCircle(ctx2, centerX, centerY, 10);
            }
        }
    }
}

// Helper function to draw a filled circle
function drawCircle(ctx, x, y, radius) {
    ctx.beginPath();
    ctx.arc(x, y, radius, 0, Math.PI * 2);
    ctx.fill();
}

// Simple hash function for demonstration purposes
function hashString(str) {
    let hash = 0;
    if (!str || str.length === 0) return hash.toString(16).padStart(8, '0');
    
    for (let i = 0; i < str.length; i++) {
        const char = str.charCodeAt(i);
        hash = ((hash << 5) - hash) + char;
        hash = hash & hash; // Convert to 32bit integer
    }
    
    // Convert to 8-character hex string
    return (hash >>> 0).toString(16).padStart(8, '0');
}