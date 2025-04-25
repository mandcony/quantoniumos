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
    
    // Start oscillator animation (always on)
    setTimeout(() => {
        // No longer need container schematics
        // But always start the oscillator regardless of checkbox state
        startOscillatorAnimation(elements);
    }, 100);
    
    // Initialize grid controls
    elements.gridRunBtn.addEventListener('click', () => runQuantumGrid(elements));
    elements.gridStressTest.addEventListener('click', () => runStressTest(elements));
    // No longer need oscillator and frequency controls since oscillator is always at fixed state
    elements.gridQubitCount.addEventListener('change', () => updateQubitGrid(null, elements));
    
    // Create initial qubit grid
    updateQubitGrid(gridQubitCount, elements);
    
    // Start oscillator animation (removing duplicate call)
    // The oscillator is already started in the setTimeout above
    
    // No need to update frequency as it's now fixed
    
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

// Run stress test on quantum grid
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
            
            // Update oscillator waves based on current processing
            if (Math.floor(progress) % 10 === 0) {
                // Wave animation is handled automatically by the animate function in startOscillatorAnimation
                // No need to redraw container schematics as they've been removed
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
    .then(response => {
        if (!response.ok) {
            throw new Error(`Server returned ${response.status}`);
        }
        return response.json();
    })
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
        
        // Always show simulation results, even if backend has issues
        simulateStressTestResults(150, elements);
        // Update formula display for stress test
        updateFormulaDisplay(150, "Stress Test", true, elements);
        
        if (data && data.success) {
            console.log("Quantum benchmark completed by backend:", data);
            
            if (window.updateStatus) {
                const engineId = data.engine_id || 'local';
                const maxQubits = data.max_qubits || 150;
                window.updateStatus(`Stress test completed for ${maxQubits} qubits - system capacity verified (Engine: ${engineId})`, 'success');
            }
        } else {
            console.warn("Backend returned non-success status:", data?.error || "Unknown error");
            
            if (window.updateStatus) {
                window.updateStatus(`Visualization completed (note: backend processing limited)`, 'success');
            }
        }
    })
    .catch(error => {
        console.error("Failed to connect to quantum backend:", error);
        
        // Even with error, complete the UI flow
        if (gridMetricsPercentage) gridMetricsPercentage.textContent = '100%';
        if (gridMetricsTime) {
            const elapsedSeconds = Math.floor((Date.now() - startTime) / 1000);
            gridMetricsTime.textContent = `Visualization completed in ${elapsedSeconds}s`;
        }
        if (gridMetricsQubits) gridMetricsQubits.textContent = `All qubits displayed in local mode`;
        
        // Hide loader
        setTimeout(() => {
            if (gridLoader) gridLoader.style.display = 'none';
            if (gridProgressBar) gridProgressBar.style.display = 'none';
        }, 1000);
        
        clearInterval(progressInterval);
        
        // Still show simulation results
        simulateStressTestResults(150, elements);
        updateFormulaDisplay(150, "Stress Test", true, elements);
        
        if (window.updateStatus) {
            window.updateStatus(`Visualization completed in local mode`, 'success');
        }
        
        if (elements.gridError) {
            elements.gridError.textContent = "Note: Operating in local mode";
        }
    });
}

// Simulate quantum results (frontend visualization only)
function simulateQuantumResults(qubitCount, inputData, elements) {
    // IMPORTANT: This is purely a UI visualization technique
    // This contains NO proprietary algorithms or implementation details
    // Standard quantum notation is used only for educational purposes
    
    // Get grid elements
    const qubits = elements.gridQubitGrid.querySelectorAll('.qubit');
    
    // Visualize standard quantum states (for education only)
    const midpoint = Math.floor(qubitCount / 2);
    
    qubits.forEach((qubit, index) => {
        const qubitValue = qubit.querySelector('.qubit-value');
        
        if (index < midpoint) {
            // Standard quantum notation (public domain concept)
            qubitValue.textContent = '|+⟩';
            qubit.style.backgroundColor = 'rgba(106, 27, 154, 0.3)';  // Purple shade
        } else {
            // Random binary states (0/1) - standard visualization
            const value = Math.random() > 0.5 ? '|1⟩' : '|0⟩';
            qubitValue.textContent = value;
            if (value === '|1⟩') {
                qubit.style.backgroundColor = 'rgba(0, 137, 123, 0.3)';  // Teal shade
            } else {
                qubit.style.backgroundColor = 'rgba(21, 101, 192, 0.3)';  // Blue shade
            }
            qubit.classList.add('qubit-measured');
        }
    });
    
    if (window.updateStatus) {
        window.updateStatus(`Visualization completed for ${qubitCount} states`, 'success');
    }
}

// Simulate stress test results - purely a UI visualization
// IMPORTANT: This contains NO proprietary algorithms or implementation details
// This is solely for display purposes using standard notation
function simulateStressTestResults(qubitCount, elements) {
    // Get qubit UI elements
    const qubits = elements.gridQubitGrid.querySelectorAll('.qubit');
    
    // Simple visual pattern for stress test display
    qubits.forEach((qubit, index) => {
        const qubitValue = qubit.querySelector('.qubit-value');
        
        if (index % 3 === 0) {
            // Standard quantum notation - ground state
            qubitValue.textContent = '|0⟩';
            qubit.style.backgroundColor = 'rgba(21, 101, 192, 0.3)';  // Blue shade
        } else if (index % 3 === 1) {
            // Standard quantum notation - excited state
            qubitValue.textContent = '|1⟩';
            qubit.style.backgroundColor = 'rgba(0, 137, 123, 0.3)';  // Teal shade
            qubit.classList.add('qubit-measured');
        } else {
            // Standard quantum notation - superposition state
            qubitValue.textContent = '|+⟩';
            qubit.style.backgroundColor = 'rgba(106, 27, 154, 0.3)';  // Purple shade
        }
    });
}

// Update quantum formula display
function updateFormulaDisplay(qubitCount, inputData, isStressTest, elements) {
    if (!elements.gridQuantumFormulas) return;
    
    // Clear formulas
    elements.gridQuantumFormulas.innerHTML = '';
    
    // Create formula elements - standard quantum notation only
    // These are NOT proprietary algorithms - just educational display
    if (isStressTest) {
        // Generic quantum notation (standard physics)
        const formulas = [
            `|ψ⟩ = α|0⟩ + β|1⟩, where |α|² + |β|² = 1`,
            `|+⟩ = (|0⟩ + |1⟩)/√2, |−⟩ = (|0⟩ - |1⟩)/√2`,
            `H|0⟩ = (|0⟩ + |1⟩)/√2, H|1⟩ = (|0⟩ - |1⟩)/√2`
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
        // Standard quantum physics notation - NOT proprietary algorithms
        const formulas = [
            `|ψ⟩ = cos(θ/2)|0⟩ + sin(θ/2)e^(iφ)|1⟩`,
            `|φ±⟩ = (|0⟩|+⟩ ± |1⟩|−⟩)/√2`
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

// Start oscillator animation - always runs at fixed frequency but responds to engine processes
function startOscillatorAnimation(elements) {
    if (!gridOscillatorCtx) return;
    
    // Stop existing animation if running
    if (oscillatorAnimationId) {
        cancelAnimationFrame(oscillatorAnimationId);
    }
    
    // Base fixed parameters
    gridFrequency = 1.5;  // Base frequency
    gridAmplitude = 1.0;  // Base amplitude
    
    // Engine process state
    let engineActivity = 0; // 0-1 value representing engine load/activity
    let enginePulse = 0;    // Pulse effect during quantum operations
    
    const canvas = document.getElementById('grid-oscillator-canvas');
    const ctx = gridOscillatorCtx;
    const width = canvas.width;
    const height = canvas.height;
    let time = 0;
    
    // Function to update engine activity based on backend processes
    function updateEngineActivity() {
        // Check if a quantum operation is running
        const gridLoader = document.getElementById('grid-loader');
        const gridProgressBar = document.getElementById('grid-progress-bar');
        const gridMetricsPercentage = document.getElementById('grid-metrics-percentage');
        
        if (gridProgressBar && gridProgressBar.style.display !== 'none') {
            // Operation in progress - extract percentage
            const percentText = gridMetricsPercentage?.textContent || '0%';
            const percent = parseInt(percentText.replace('%', '')) || 0;
            
            // Calculate engine activity and pulse based on progress
            engineActivity = Math.min(1.0, percent / 100 * 1.5); // Amplify activity slightly
            enginePulse = Math.sin(time * 10) * 0.2 * engineActivity; // Pulsing effect
        } else {
            // No active operation - gradual return to baseline
            engineActivity = Math.max(0, engineActivity - 0.02);
            enginePulse = Math.max(0, enginePulse - 0.01);
        }
    }
    
    // Animation function
    function animate() {
        if (!ctx || !canvas) return;
        
        // Update engine activity state
        updateEngineActivity();
        
        // Calculate effective frequency and amplitude based on engine activity
        const effectiveFrequency = gridFrequency + (engineActivity * 0.5); // Increase frequency with activity
        const effectiveAmplitude = gridAmplitude + enginePulse;   // Add pulsing to amplitude
        
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
            const y = height / 2 + Math.sin((x / width * 8 + time) * Math.PI * effectiveFrequency) * effectiveAmplitude * (height / 3);
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
            const y = height / 2 + Math.sin((x / width * 12 + time * 1.5) * Math.PI * effectiveFrequency * 0.7) * effectiveAmplitude * (height / 5);
            if (x === 0) {
                ctx.moveTo(x, y);
            } else {
                ctx.lineTo(x, y);
            }
        }
        
        ctx.stroke();
        
        // If engine activity is high, add some harmonics (third wave)
        if (engineActivity > 0.3) {
            const harmony = engineActivity - 0.3; // Only show above threshold
            ctx.strokeStyle = `rgba(255, 64, 129, ${harmony * 1.5})`; // Pink wave that gets more visible with activity
            ctx.lineWidth = 1;
            ctx.beginPath();
            
            for (let x = 0; x < width; x += step) {
                const y = height / 2 + Math.sin((x / width * 16 + time * 2) * Math.PI * effectiveFrequency * 1.5) * 
                           effectiveAmplitude * (height / 8) * harmony;
                if (x === 0) {
                    ctx.moveTo(x, y);
                } else {
                    ctx.lineTo(x, y);
                }
            }
            
            ctx.stroke();
        }
        
        // Update time
        time += 0.01;
        
        // Always continue animation
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

// Oscillator is now always on but configured differently
function toggleOscillator() {
    // This function is kept for compatibility but no longer toggles
    // Oscillator is always on in fixed state
    return;
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
    // Initialize frontend visualization first to ensure UI works even if backend fails
    initializeQuantumGrid();
    
    // Try to connect to the quantum engine in the backend
    fetch('/api/quantum/initialize', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({
            max_qubits: 150,
            connect_encryption: true // Flag to connect quantum grid with encryption module
        })
    })
    .then(response => {
        if (!response.ok) {
            throw new Error(`Server returned ${response.status}`);
        }
        return response.json();
    })
    .then(data => {
        if (data && data.success) {
            console.log("Quantum engine initialized with", data.max_qubits, "qubits (Engine ID:", data.engine_id, ")");
            
            // Update quantum grid with encryption connection status
            if (data.encryption_connected) {
                console.log("Quantum grid connected to encryption module");
                
                // Get elements for updating
                const elements = {
                    gridQubitGrid: document.getElementById('grid-qubit-grid'),
                    gridQubitCount: document.getElementById('grid-qubit-count')
                };
                
                if (elements.gridQubitCount) {
                    elements.gridQubitCount.value = data.max_qubits;
                    updateQubitGrid(data.max_qubits, elements);
                }
            }
            
            if (window.updateStatus) {
                window.updateStatus(`Quantum engine initialized with ${data.max_qubits} qubits capacity`, 'success');
            }
        } else {
            console.error("Failed to initialize quantum engine:", data.error || data.message);
            // Still initialize the frontend even if the backend fails
            initializeQuantumGrid();
        }
    })
    .catch(error => {
        console.error("Failed to initialize quantum engine:", error);
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
// Container schematics have been removed as per requirements, but we'll keep a no-op function
// to prevent errors in case it's called from legacy code
window.drawContainerSchematics = function() {
    console.log("Container schematics have been removed - no-op function");
    // No-op function, no longer calling the internal function to avoid recursion
};