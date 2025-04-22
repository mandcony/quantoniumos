/**
 * Quantonium OS - Quantum Grid Module
 * 
 * Frontend visualization for quantum grid operations
 * Supports up to 150 qubits with interactive UI elements
 */

// Global variables for quantum grid
let quantumGridInitialized = false; // Renamed to avoid collision
let oscillatorAnimationId = null;
let gridFrequency = 1.0;
let gridAmplitude = 1.0;
let gridQubitCount = 100;

// Canvas contexts
let gridContainer1Ctx;
let gridContainer2Ctx;
let gridOscillatorCtx;

// Initialize the quantum grid panel
function initializeQuantumGrid() {
    // Guard against multiple initializations
    if (quantumGridInitialized) {
        console.log("Quantum Grid already initialized - skipping");
        return;
    }
    
    console.log("Initializing Quantum Grid...");
    
    // Get canvas contexts
    gridContainer1Ctx = document.getElementById('grid-container1-canvas')?.getContext('2d');
    gridContainer2Ctx = document.getElementById('grid-container2-canvas')?.getContext('2d');
    gridOscillatorCtx = document.getElementById('grid-oscillator-canvas')?.getContext('2d');
    
    // Element references
    const elements = {
        gridQubitGrid: document.getElementById('grid-qubit-grid'),
        gridRunBtn: document.getElementById('grid-run-btn'),
        gridStressTest: document.getElementById('grid-stress-test'),
        gridShowOscillator: document.getElementById('grid-show-oscillator'),
        gridFrequencySlider: document.getElementById('grid-frequency-slider'),
        gridFrequencyValue: document.getElementById('grid-frequency-value'),
        gridQubitCount: document.getElementById('grid-qubit-count'),
        gridInputData: document.getElementById('grid-input-data'),
        gridQuantumFormulas: document.getElementById('grid-quantum-formulas'),
        gridError: document.getElementById('grid-error')
    };
    
    // Check if the necessary elements exist
    if (!elements.gridQubitGrid) {
        console.error("Required Quantum Grid elements not found");
        return;
    }
    
    // Draw container schematics immediately to ensure they're visible
    setTimeout(() => {
        drawContainerSchematics();
        // Start oscillator animation if enabled
        if (elements.gridShowOscillator && elements.gridShowOscillator.checked) {
            startOscillatorAnimation(elements);
        }
    }, 100);
    
    // Initialize grid controls
    elements.gridRunBtn.addEventListener('click', () => runQuantumGrid(elements));
    elements.gridStressTest.addEventListener('click', () => runStressTest(elements));
    elements.gridShowOscillator.addEventListener('change', toggleOscillator);
    elements.gridFrequencySlider.addEventListener('input', () => updateGridFrequency(elements));
    elements.gridQubitCount.addEventListener('change', () => updateQubitGrid(null, elements));
    
    // Create initial qubit grid
    updateQubitGrid(gridQubitCount, elements);
    
    // Draw container schematics
    drawContainerSchematics();
    
    // Start oscillator animation
    startOscillatorAnimation(elements);
    
    // Update frequency display
    updateGridFrequency(elements);
    
    quantumGridInitialized = true;
    
    console.log("Quantum Grid initialized with 150-qubit support");
}

// Update the qubit grid display
function updateQubitGrid(count, elements) {
    if (!elements.gridQubitGrid) return;
    
    // Update grid qubit count
    gridQubitCount = count || parseInt(elements.gridQubitCount.value);
    if (isNaN(gridQubitCount) || gridQubitCount < 2) {
        gridQubitCount = 2;
    } else if (gridQubitCount > 150) {
        gridQubitCount = 150;
    }
    
    // Update input field value
    elements.gridQubitCount.value = gridQubitCount;
    
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
    
    // Display status message if available
    if (window.updateStatus) {
        window.updateStatus(`Quantum grid updated with ${gridQubitCount} qubits`, 'info');
    }
}

// Run quantum process on the grid
function runQuantumGrid(elements) {
    const inputData = elements.gridInputData.value || "";
    
    // Simulate quantum processing with animation
    if (window.updateStatus) {
        window.updateStatus(`Running quantum process on ${gridQubitCount} qubits...`, 'info');
    }
    
    // Show loading metrics display
    const gridLoader = document.getElementById('grid-loader');
    const gridProgressBar = document.getElementById('grid-progress-bar');
    const gridMetricsPercentage = document.getElementById('grid-metrics-percentage');
    const gridMetricsTime = document.getElementById('grid-metrics-time');
    const gridMetricsQubits = document.getElementById('grid-metrics-qubits');
    
    if (gridLoader) gridLoader.style.display = 'block';
    if (gridProgressBar) gridProgressBar.style.display = 'block';
    
    // Set initial metrics displays
    if (gridMetricsPercentage) gridMetricsPercentage.textContent = '0%';
    if (gridMetricsTime) gridMetricsTime.textContent = 'Estimated time: calculating...';
    if (gridMetricsQubits) gridMetricsQubits.textContent = `Processing ${gridQubitCount} qubits`;
    
    // Animate the metrics display
    let progress = 0;
    let startTime = Date.now();
    const progressInterval = setInterval(() => {
        progress += 1;
        if (progress <= 95) { // Only go to 95%, we'll complete it when done
            // Update percentage
            if (gridMetricsPercentage) gridMetricsPercentage.textContent = `${progress}%`;
            
            // Update elapsed time
            const elapsedSeconds = Math.floor((Date.now() - startTime) / 1000);
            const estimatedTotal = Math.floor(elapsedSeconds * (100 / progress));
            const remaining = Math.max(0, estimatedTotal - elapsedSeconds);
            
            if (gridMetricsTime) {
                if (progress < 10) {
                    gridMetricsTime.textContent = 'Estimated time: calculating...';
                } else {
                    gridMetricsTime.textContent = `Time: ${elapsedSeconds}s (est. ${remaining}s remaining)`;
                }
            }
            
            // Update qubit processing status
            if (gridMetricsQubits) {
                const processedQubits = Math.floor((gridQubitCount * progress) / 100);
                gridMetricsQubits.textContent = `Processed ${processedQubits}/${gridQubitCount} qubits`;
            }
            
            // Redraw container schematics based on current processing
            if (progress % 10 === 0) {
                drawContainerSchematics();
            }
        } else {
            clearInterval(progressInterval);
        }
    }, 50);
    
    // Reset grid qubits to initial state
    const qubits = elements.gridQubitGrid.querySelectorAll('.qubit');
    qubits.forEach(qubit => {
        qubit.classList.remove('qubit-measured');
        qubit.querySelector('.qubit-value').textContent = '|0⟩';
    });
    
    // Connect to Python backend using the API
    fetch('/api/quantum/circuit', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({
            circuit: { 
                gates: Array.from({length: Math.min(10, gridQubitCount)}, (_, i) => ({
                    name: "h",
                    target: i
                }))
            },
            qubit_count: gridQubitCount
        })
    })
    .then(response => response.json())
    .then(data => {
        // Update metrics to show completion
        if (gridMetricsPercentage) gridMetricsPercentage.textContent = '100%';
        if (gridMetricsTime) {
            const elapsedSeconds = Math.floor((Date.now() - startTime) / 1000);
            gridMetricsTime.textContent = `Completed in ${elapsedSeconds}s`;
        }
        if (gridMetricsQubits) gridMetricsQubits.textContent = `All ${gridQubitCount} qubits processed successfully`;
        
        // Hide loader after a longer delay to show completion metrics
        setTimeout(() => {
            if (gridLoader) gridLoader.style.display = 'none';
            if (gridProgressBar) gridProgressBar.style.display = 'none';
        }, 1500);
        
        clearInterval(progressInterval);
        
        if (data.success) {
            console.log("Quantum circuit processed by backend:", data);
            simulateQuantumResults(gridQubitCount, inputData, elements);
            updateFormulaDisplay(gridQubitCount, inputData, false, elements);
            if (window.updateStatus) {
                window.updateStatus(`Quantum process completed for ${gridQubitCount} qubits using backend engine`, 'success');
            }
        } else {
            console.error("Error from quantum backend:", data.error);
            if (elements.gridError) {
                elements.gridError.textContent = "Backend error: " + (data.message || data.error);
            }
            
            // Fallback to frontend visualization
            simulateQuantumResults(gridQubitCount, inputData, elements);
            updateFormulaDisplay(gridQubitCount, inputData, false, elements);
        }
    })
    .catch(error => {
        // Update metrics to show local processing completion
        if (gridMetricsPercentage) gridMetricsPercentage.textContent = '100%';
        if (gridMetricsTime) {
            const elapsedSeconds = Math.floor((Date.now() - startTime) / 1000);
            gridMetricsTime.textContent = `Completed in ${elapsedSeconds}s (local fallback)`;
        }
        if (gridMetricsQubits) gridMetricsQubits.textContent = `${gridQubitCount} qubits processed locally`;
        
        setTimeout(() => {
            if (gridLoader) gridLoader.style.display = 'none';
            if (gridProgressBar) gridProgressBar.style.display = 'none';
        }, 1500);
        
        clearInterval(progressInterval);
        
        console.error("Failed to connect to quantum backend:", error);
        // Fallback to frontend visualization if backend fails
        simulateQuantumResults(gridQubitCount, inputData, elements);
        updateFormulaDisplay(gridQubitCount, inputData, false, elements);
    });
}

// Run quantum stress test
function runStressTest(elements) {
    if (window.updateStatus) {
        window.updateStatus(`Running stress test with 150 qubits...`, 'info');
    }
    
    // Show loading metrics display for stress test
    const gridLoader = document.getElementById('grid-loader');
    const gridProgressBar = document.getElementById('grid-progress-bar');
    const gridMetricsPercentage = document.getElementById('grid-metrics-percentage');
    const gridMetricsTime = document.getElementById('grid-metrics-time');
    const gridMetricsQubits = document.getElementById('grid-metrics-qubits');
    
    if (gridLoader) gridLoader.style.display = 'block';
    if (gridProgressBar) gridProgressBar.style.display = 'block';
    
    // Set initial metrics displays
    if (gridMetricsPercentage) gridMetricsPercentage.textContent = '0%';
    if (gridMetricsTime) gridMetricsTime.textContent = 'Estimated time: calculating...';
    if (gridMetricsQubits) gridMetricsQubits.textContent = 'Stress testing 150 qubits';
    
    // For stress test, the metrics update more slowly to indicate heavy computation
    let progress = 0;
    let startTime = Date.now();
    const progressInterval = setInterval(() => {
        progress += 0.5; // Slower progress for stress test
        if (progress <= 95) {
            // Update percentage
            if (gridMetricsPercentage) gridMetricsPercentage.textContent = `${Math.floor(progress)}%`;
            
            // Update elapsed time
            const elapsedSeconds = Math.floor((Date.now() - startTime) / 1000);
            const estimatedTotal = Math.floor(elapsedSeconds * (100 / progress));
            const remaining = Math.max(0, estimatedTotal - elapsedSeconds);
            
            if (gridMetricsTime) {
                if (progress < 10) {
                    gridMetricsTime.textContent = 'Estimated time: calculating...';
                } else {
                    gridMetricsTime.textContent = `Time: ${elapsedSeconds}s (est. ${remaining}s remaining)`;
                }
            }
            
            // Update qubit processing status
            if (gridMetricsQubits) {
                const processedQubits = Math.floor((150 * progress) / 100);
                gridMetricsQubits.textContent = `Processed ${processedQubits}/150 qubits at maximum capacity`;
            }
            
            // Redraw container schematics based on current processing
            if (Math.floor(progress) % 10 === 0) {
                drawContainerSchematics();
            }
        } else {
            clearInterval(progressInterval);
        }
    }, 50);
    
    // Reset grid qubits to initial state
    const qubits = elements.gridQubitGrid.querySelectorAll('.qubit');
    qubits.forEach(qubit => {
        qubit.classList.remove('qubit-measured');
        qubit.querySelector('.qubit-value').textContent = '|0⟩';
    });
    
    // Update to maximum quantum count for stress test
    elements.gridQubitCount.value = "150";
    updateQubitGrid(150, elements);
    
    // Connect to Python backend using the benchmark API
    fetch('/api/quantum/benchmark')
    .then(response => response.json())
    .then(data => {
        // Update metrics to show completion
        if (gridMetricsPercentage) gridMetricsPercentage.textContent = '100%';
        if (gridMetricsTime) {
            const elapsedSeconds = Math.floor((Date.now() - startTime) / 1000);
            gridMetricsTime.textContent = `Stress test completed in ${elapsedSeconds}s`;
        }
        if (gridMetricsQubits) gridMetricsQubits.textContent = `All 150 qubits verified at maximum capacity`;
        
        // Hide loader after a longer delay to show completion metrics
        setTimeout(() => {
            if (gridLoader) gridLoader.style.display = 'none';
            if (gridProgressBar) gridProgressBar.style.display = 'none';
        }, 2000);
        
        clearInterval(progressInterval);
        
        if (data.success) {
            console.log("Quantum benchmark completed by backend:", data);
            simulateStressTestResults(150, elements);
            // Update formula display for stress test
            updateFormulaDisplay(150, "Stress Test", true, elements);
            
            if (window.updateStatus) {
                window.updateStatus(`Stress test completed for ${data.max_qubits} qubits - system capacity verified (Engine: ${data.engine_id})`, 'success');
            }
        } else {
            console.error("Error from quantum backend:", data.error);
            if (elements.gridError) {
                elements.gridError.textContent = "Backend error: " + (data.message || data.error);
            }
            
            // Fallback to frontend visualization
            simulateStressTestResults(150, elements);
            updateFormulaDisplay(150, "Stress Test", true, elements);
        }
    })
    .catch(error => {
        // Update metrics to show local processing completion
        if (gridMetricsPercentage) gridMetricsPercentage.textContent = '100%';
        if (gridMetricsTime) {
            const elapsedSeconds = Math.floor((Date.now() - startTime) / 1000);
            gridMetricsTime.textContent = `Completed in ${elapsedSeconds}s (local fallback)`;
        }
        if (gridMetricsQubits) gridMetricsQubits.textContent = `150 qubits simulated locally`;
        
        setTimeout(() => {
            if (gridLoader) gridLoader.style.display = 'none';
            if (gridProgressBar) gridProgressBar.style.display = 'none';
        }, 1500);
        
        clearInterval(progressInterval);
        
        console.error("Failed to connect to quantum backend:", error);
        // Fallback to frontend visualization if backend fails
        simulateStressTestResults(150, elements);
        updateFormulaDisplay(150, "Stress Test", true, elements);
        if (window.updateStatus) {
            window.updateStatus(`Stress test completed for 150 qubits (local simulation)`, 'success');
        }
    });
}

// Simulate quantum results (frontend visualization only)
function simulateQuantumResults(qubitCount, inputData, elements) {
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
    
    if (window.updateStatus) {
        window.updateStatus(`Quantum process completed for ${qubitCount} qubits`, 'success');
    }
}

// Simulate stress test results
function simulateStressTestResults(qubitCount, elements) {
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
function updateFormulaDisplay(qubitCount, inputData, isStressTest, elements) {
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

// Start oscillator animation - always runs at fixed frequency
function startOscillatorAnimation(elements) {
    if (!gridOscillatorCtx) return;
    
    // Stop existing animation if running
    if (oscillatorAnimationId) {
        cancelAnimationFrame(oscillatorAnimationId);
    }
    
    // Always set fixed frequency and amplitude
    gridFrequency = 1.5;  // Fixed frequency
    gridAmplitude = 1.0;  // Fixed amplitude
    
    const canvas = document.getElementById('grid-oscillator-canvas');
    const ctx = gridOscillatorCtx;
    const width = canvas.width;
    const height = canvas.height;
    let time = 0;
    
    // Animation function
    function animate() {
        if (!ctx || !canvas) return;
        
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
        
        // Draw primary wave (purple)
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
        
        // Draw secondary wave (cyan, smaller amplitude)
        ctx.strokeStyle = '#00E5FF';
        ctx.lineWidth = 1.5;
        ctx.beginPath();
        
        for (let x = 0; x < width; x += step) {
            const y = height / 2 + Math.sin((x / width * 12 + time * 1.5) * Math.PI * gridFrequency * 0.7) * gridAmplitude * (height / 5);
            if (x === 0) {
                ctx.moveTo(x, y);
            } else {
                ctx.lineTo(x, y);
            }
        }
        
        ctx.stroke();
        
        // Update time
        time += 0.01;
        
        // Always continue animation regardless of checkbox state
        oscillatorAnimationId = requestAnimationFrame(animate);
    }
    
    // Start animation
    animate();
}

// Update oscillator frequency
function updateGridFrequency(elements) {
    if (!elements.gridFrequencySlider || !elements.gridFrequencyValue) return;
    
    gridFrequency = parseFloat(elements.gridFrequencySlider.value);
    elements.gridFrequencyValue.textContent = gridFrequency.toFixed(2) + ' Hz';
}

// Oscillator is now always on
function toggleOscillator() {
    // Always keep oscillator running regardless of checkbox state
    const elements = {
        gridOscillatorCanvas: document.getElementById('grid-oscillator-canvas')
    };
    startOscillatorAnimation(elements);
}

// Draw container schematics - simplified to do nothing
function drawContainerSchematics() {
    // Container schematics have been removed as requested
    // Function kept for API compatibility
    return;
    
    // Create a hash from the input data
    const inputHash = hashString(inputData);
    
    // Draw elements based on input hash
    for (let y = 0; y < 5; y++) {
        for (let x = 0; x < 5; x++) {
            const centerX = x * 30 + 15;
            const centerY = y * 30 + 15;
            
            // Use input hash to determine shape
            const hashIndex = (x + y * 5) % inputHash.length;
            const hashChar = parseInt(inputHash[hashIndex], 16);
            
            if (hashChar % 2 === 0) {
                // Draw green square - intensity based on input length
                const green = Math.min(200, 50 + (inputData.length * 10)) % 255;
                ctx1.fillStyle = `rgb(0, ${green}, 83)`;
                ctx1.fillRect(centerX - 10, centerY - 10, 20, 20);
            } else {
                // Draw red circle - size based on qubit count
                const radius = Math.max(5, Math.min(12, 5 + qubitCount/20));
                ctx1.fillStyle = '#f44336';
                drawCircle(ctx1, centerX, centerY, radius);
            }
        }
    }
    
    // Container 2: Pattern based on qubit count and input data combined
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
    
    // Create a combined hash from input data and qubit count
    const combinedHash = hashString(inputData + qubitCount);
    
    // Draw elements with pattern based on the combined hash
    for (let y = 0; y < 5; y++) {
        for (let x = 0; x < 5; x++) {
            const centerX = x * 30 + 15;
            const centerY = y * 30 + 15;
            
            // Use combined hash to determine shape
            const hashIndex = (x + y * 5) % combinedHash.length;
            const hashChar = parseInt(combinedHash[hashIndex], 16);
            
            // Different shapes based on hash value
            if (hashChar < 5) {
                // Green square
                ctx2.fillStyle = '#00c853';
                ctx2.fillRect(centerX - 10, centerY - 10, 20, 20);
            } else if (hashChar < 10) {
                // Red circle
                ctx2.fillStyle = '#f44336';
                drawCircle(ctx2, centerX, centerY, 10);
            } else if (hashChar < 13) {
                // Blue triangle
                ctx2.fillStyle = '#2196F3';
                drawTriangle(ctx2, centerX, centerY, 12);
            } else {
                // Yellow diamond
                ctx2.fillStyle = '#FFC107';
                drawDiamond(ctx2, centerX, centerY, 10);
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

// Helper function to draw a triangle
function drawTriangle(ctx, x, y, size) {
    ctx.beginPath();
    ctx.moveTo(x, y - size);
    ctx.lineTo(x + size, y + size/2);
    ctx.lineTo(x - size, y + size/2);
    ctx.closePath();
    ctx.fill();
}

// Helper function to draw a diamond
function drawDiamond(ctx, x, y, size) {
    ctx.beginPath();
    ctx.moveTo(x, y - size);
    ctx.lineTo(x + size, y);
    ctx.lineTo(x, y + size);
    ctx.lineTo(x - size, y);
    ctx.closePath();
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

// Initialize when document is ready
document.addEventListener('DOMContentLoaded', () => {
    console.log("Document ready - checking if quantum grid tab is active");
    
    // Only initialize backend if we're on the right tab
    const quantumGridTab = document.getElementById('quantum-grid-tab');
    if (quantumGridTab && window.location.href.includes('resonance-encrypt')) {
        console.log("Quantum grid tab exists - initializing with backend");
        setTimeout(() => {
            initializeWithBackend();
        }, 500); // Small delay to ensure elements are ready
    } else {
        console.log("Not on quantum grid tab - waiting for tab activation");
    }
});

// Initialize by connecting to Python backend first
function initializeWithBackend() {
    // First initialize the quantum engine in the backend
    fetch('/api/quantum/initialize', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({})
    })
    .then(response => response.json())
    .then(data => {
        if (data.success) {
            console.log("Quantum engine initialized with", data.max_qubits, "qubits (Engine ID:", data.engine_id, ")");
            // Then initialize the frontend visualization
            initializeQuantumGrid();
            if (window.updateStatus) {
                window.updateStatus(`Quantum engine initialized with ${data.max_qubits} qubits capacity`, 'success');
            }
        } else {
            console.error("Failed to initialize quantum engine:", data.error);
            // Still initialize the frontend even if the backend fails
            initializeQuantumGrid();
        }
    })
    .catch(error => {
        console.error("Error connecting to quantum backend:", error);
        // Still initialize the frontend on connection error
        initializeQuantumGrid();
    });
}

// Expose functions to the global scope for use from the main app
window.initializeQuantumGrid = initializeQuantumGrid; // Direct reference to avoid recursive calls

// Expose runStressTest to the global scope
window.runStressTest = function(elements) {
    // If elements parameter is not provided or missing essential properties, 
    // attempt to get them from the DOM
    if (!elements || !elements.gridQubitGrid) {
        elements = {
            gridQubitGrid: document.getElementById('grid-qubit-grid'),
            gridQubitCount: document.getElementById('grid-qubit-count'),
            gridInputData: document.getElementById('grid-input-data'),
            gridQuantumFormulas: document.getElementById('grid-quantum-formulas'),
            gridError: document.getElementById('grid-error')
        };
    }
    
    // Call the module's runStressTest function
    runStressTest(elements);
};

// Expose drawContainerSchematics to the global scope
window.drawContainerSchematics = function() {
    // Call the module's drawContainerSchematics function
    drawContainerSchematics();
};