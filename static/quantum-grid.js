/**
 * QuantoniumOS Quantum Grid Visualization
 * 
 * Provides a visual representation of the quantum grid with 150 qubits
 * and their entanglement patterns during encryption operations.
 */

// Maintain global state for grid visualizations
const quantumGridState = {
    oscillators: {},
    grid1Canvas: {},
    grid2Canvas: {},
    animationFrames: {},
    points: [],
    connections: [],
    oscillatorPoints: [],
    activeOscillators: new Set(),
    qubitCount: 150, // Default qubit count
    initialized: false
};

// Initialize quantum grid visualizations for a specific window
function initializeQuantumGrid(windowId) {
    // Set up grid canvases
    const grid1Canvas = document.getElementById(`grid-container1-canvas-${windowId}`);
    const grid2Canvas = document.getElementById(`grid-container2-canvas-${windowId}`);
    const oscillatorCanvas = document.getElementById(`grid-oscillator-canvas-${windowId}`);
    
    if (!grid1Canvas || !grid2Canvas || !oscillatorCanvas) {
        console.error('Grid canvases not found for window:', windowId);
        return;
    }
    
    // Store canvases in state
    quantumGridState.grid1Canvas[windowId] = grid1Canvas;
    quantumGridState.grid2Canvas[windowId] = grid2Canvas;
    quantumGridState.oscillators[windowId] = oscillatorCanvas;
    
    // Generate initial points for qubits (75 per grid)
    generateQuantumGridPoints();
    
    // Setup event listeners for controls
    setupGridControls(windowId);
    
    // Initial drawing of grids
    drawQuantumGrid(windowId);
    
    // Start oscillator animation if it's set to active
    if (!quantumGridState.animationFrames[windowId]) {
        startOscillatorAnimation(windowId);
    }
    
    // Mark as initialized
    quantumGridState.initialized = true;
}

// Generate points for quantum grid visualization
function generateQuantumGridPoints() {
    // Only generate points if not already done
    if (quantumGridState.points.length > 0) {
        return;
    }
    
    // Generate points for first grid (qubits 1-75)
    const grid1Points = [];
    for (let i = 0; i < 75; i++) {
        grid1Points.push({
            x: Math.random() * 400,
            y: Math.random() * 200,
            radius: 2 + Math.random() * 2,
            hue: i / 75 * 360,
            connections: []
        });
    }
    
    // Generate points for second grid (qubits 76-150)
    const grid2Points = [];
    for (let i = 0; i < 75; i++) {
        grid2Points.push({
            x: Math.random() * 400,
            y: Math.random() * 200,
            radius: 2 + Math.random() * 2,
            hue: (i + 75) / 150 * 360,
            connections: []
        });
    }
    
    // Generate random connections (entanglements) between qubits
    const connections = [];
    
    // Within grid 1
    for (let i = 0; i < 30; i++) {
        const source = Math.floor(Math.random() * 75);
        const target = Math.floor(Math.random() * 75);
        if (source !== target) {
            connections.push({
                source: source,
                target: target,
                grid: 1,
                strength: 0.3 + Math.random() * 0.7
            });
            grid1Points[source].connections.push(target);
            grid1Points[target].connections.push(source);
        }
    }
    
    // Within grid 2
    for (let i = 0; i < 30; i++) {
        const source = Math.floor(Math.random() * 75);
        const target = Math.floor(Math.random() * 75);
        if (source !== target) {
            connections.push({
                source: source,
                target: target,
                grid: 2,
                strength: 0.3 + Math.random() * 0.7
            });
            grid2Points[source].connections.push(target);
            grid2Points[target].connections.push(source);
        }
    }
    
    // Between grids (cross-grid entanglement)
    for (let i = 0; i < 15; i++) {
        const source = Math.floor(Math.random() * 75);
        const target = Math.floor(Math.random() * 75);
        connections.push({
            source: source,
            target: target,
            grid: 3, // Code for cross-grid
            strength: 0.2 + Math.random() * 0.5
        });
    }
    
    // Generate oscillator points
    const oscillatorPoints = [];
    for (let i = 0; i < 100; i++) {
        oscillatorPoints.push({
            x: i * 8,
            y: 50,
            baseY: 50,
            amplitude: 20 + Math.random() * 10,
            frequency: 0.5 + Math.random() * 1.5,
            phase: Math.random() * Math.PI * 2
        });
    }
    
    // Store in global state
    quantumGridState.points = [...grid1Points, ...grid2Points];
    quantumGridState.connections = connections;
    quantumGridState.oscillatorPoints = oscillatorPoints;
}

// Draw the quantum grid for a specific window
function drawQuantumGrid(windowId) {
    // Draw first grid (qubits 1-75)
    drawGrid(
        quantumGridState.grid1Canvas[windowId], 
        quantumGridState.points.slice(0, 75),
        quantumGridState.connections.filter(c => c.grid === 1)
    );
    
    // Draw second grid (qubits 76-150)
    drawGrid(
        quantumGridState.grid2Canvas[windowId], 
        quantumGridState.points.slice(75, 150),
        quantumGridState.connections.filter(c => c.grid === 2)
    );
    
    // Draw cross-grid connections (if visible)
    // This is just for show - in a real implementation you'd actually visualize 
    // the cross-connections between the two canvases
    
    // Draw oscillator
    drawOscillator(windowId);
}

// Draw a single grid canvas
function drawGrid(canvas, points, connections) {
    if (!canvas) return;
    
    const ctx = canvas.getContext('2d');
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    
    // Draw connections first (so they're behind the points)
    ctx.lineWidth = 0.5;
    for (const conn of connections) {
        const source = points[conn.source];
        const target = points[conn.target];
        
        ctx.strokeStyle = `rgba(255, 255, 255, ${conn.strength * 0.5})`;
        ctx.beginPath();
        ctx.moveTo(source.x, source.y);
        ctx.lineTo(target.x, target.y);
        ctx.stroke();
    }
    
    // Draw points
    for (const point of points) {
        // Glow effect
        const gradient = ctx.createRadialGradient(
            point.x, point.y, 0,
            point.x, point.y, point.radius * 4
        );
        gradient.addColorStop(0, `rgba(255, 255, 255, 0.8)`);
        gradient.addColorStop(0.5, `rgba(200, 200, 200, 0.2)`);
        gradient.addColorStop(1, 'rgba(0, 0, 0, 0)');
        
        ctx.fillStyle = gradient;
        ctx.beginPath();
        ctx.arc(point.x, point.y, point.radius * 2, 0, Math.PI * 2);
        ctx.fill();
        
        // Actual point
        ctx.fillStyle = 'white';
        ctx.beginPath();
        ctx.arc(point.x, point.y, point.radius, 0, Math.PI * 2);
        ctx.fill();
    }
}

// Draw the oscillator visualization
function drawOscillator(windowId) {
    const canvas = quantumGridState.oscillators[windowId];
    if (!canvas) return;
    
    const ctx = canvas.getContext('2d');
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    
    // Get frequency multiplier from slider
    const frequencySlider = document.getElementById(`grid-frequency-slider-${windowId}`);
    const frequencyMultiplier = frequencySlider ? parseFloat(frequencySlider.value) : 1.0;
    
    // Draw oscillator wave
    ctx.strokeStyle = 'rgba(255, 255, 255, 0.7)';
    ctx.lineWidth = 2;
    ctx.beginPath();
    
    // Get current time for animation
    const now = Date.now() / 1000;
    
    for (let i = 0; i < quantumGridState.oscillatorPoints.length; i++) {
        const point = quantumGridState.oscillatorPoints[i];
        
        // Calculate y position with time-based animation
        if (quantumGridState.activeOscillators.has(windowId)) {
            point.y = point.baseY + Math.sin(now * point.frequency * frequencyMultiplier + point.phase) * point.amplitude;
        }
        
        if (i === 0) {
            ctx.moveTo(point.x, point.y);
        } else {
            ctx.lineTo(point.x, point.y);
        }
    }
    ctx.stroke();
    
    // Add glow effect
    ctx.strokeStyle = 'rgba(255, 255, 255, 0.3)';
    ctx.lineWidth = 4;
    ctx.stroke();
}

// Start oscillator animation for a specific window
function startOscillatorAnimation(windowId) {
    // Add to active oscillators set
    quantumGridState.activeOscillators.add(windowId);
    
    // Cancel existing animation frame if any
    if (quantumGridState.animationFrames[windowId]) {
        cancelAnimationFrame(quantumGridState.animationFrames[windowId]);
    }
    
    // Animation function
    function animate() {
        if (quantumGridState.activeOscillators.has(windowId)) {
            drawOscillator(windowId);
            quantumGridState.animationFrames[windowId] = requestAnimationFrame(animate);
        }
    }
    
    // Start animation
    animate();
}

// Stop oscillator animation for a specific window
function stopOscillatorAnimation(windowId) {
    // Remove from active oscillators
    quantumGridState.activeOscillators.delete(windowId);
    
    // Cancel animation frame
    if (quantumGridState.animationFrames[windowId]) {
        cancelAnimationFrame(quantumGridState.animationFrames[windowId]);
        quantumGridState.animationFrames[windowId] = null;
    }
    
    // Reset oscillator points to base position
    for (const point of quantumGridState.oscillatorPoints) {
        point.y = point.baseY;
    }
    
    // Redraw static oscillator
    drawOscillator(windowId);
}

// Toggle oscillator animation
function toggleOscillator(windowId) {
    if (quantumGridState.activeOscillators.has(windowId)) {
        stopOscillatorAnimation(windowId);
    } else {
        startOscillatorAnimation(windowId);
    }
}

// Setup grid control event listeners
function setupGridControls(windowId) {
    // Toggle oscillator button
    const toggleButton = document.getElementById(`grid-toggle-oscillator-${windowId}`);
    if (toggleButton) {
        toggleButton.addEventListener('click', () => {
            toggleOscillator(windowId);
        });
    }
    
    // Frequency slider
    const frequencySlider = document.getElementById(`grid-frequency-slider-${windowId}`);
    const frequencyValue = document.getElementById(`grid-frequency-value-${windowId}`);
    
    if (frequencySlider && frequencyValue) {
        frequencySlider.addEventListener('input', () => {
            const value = parseFloat(frequencySlider.value).toFixed(1);
            frequencyValue.textContent = value;
        });
    }
}

// Update grid when text or key changes to simulate quantum operations
function updateQuantumGridForEncryption(windowId, text, key) {
    if (!quantumGridState.initialized) {
        initializeQuantumGrid(windowId);
    }
    
    // Start oscillator if not already running
    if (!quantumGridState.activeOscillators.has(windowId)) {
        startOscillatorAnimation(windowId);
    }
    
    // Generate new entanglement patterns based on text and key
    updateGridEntanglements(text, key);
    
    // Redraw the grid
    drawQuantumGrid(windowId);
}

// Update entanglement patterns based on input
function updateGridEntanglements(text, key) {
    if (!text || !key || text.length < 2 || key.length < 2) {
        return;
    }
    
    // Create a simple hash from text and key
    const combinedString = text + key;
    let hash = 0;
    for (let i = 0; i < combinedString.length; i++) {
        hash = ((hash << 5) - hash) + combinedString.charCodeAt(i);
        hash |= 0;
    }
    
    // Use hash to seed pseudo-random changes to entanglement patterns
    const seededRandom = seedRandom(hash);
    
    // Update some random connections
    for (let i = 0; i < 20; i++) {
        const connectionIndex = Math.floor(seededRandom() * quantumGridState.connections.length);
        const connection = quantumGridState.connections[connectionIndex];
        
        // Update connection strength
        connection.strength = 0.2 + seededRandom() * 0.8;
    }
    
    // Update oscillator frequencies
    for (let i = 0; i < quantumGridState.oscillatorPoints.length; i++) {
        if (i % 3 === 0) { // Only change some points
            quantumGridState.oscillatorPoints[i].frequency = 0.5 + seededRandom() * 1.5;
            quantumGridState.oscillatorPoints[i].amplitude = 15 + seededRandom() * 15;
        }
    }
}

// Simple seeded random number generator
function seedRandom(seed) {
    return function() {
        seed = (seed * 9301 + 49297) % 233280;
        return seed / 233280;
    };
}

// When window is fully loaded, initialize Quantum Grid integration
document.addEventListener('DOMContentLoaded', function() {
    // This is only initialization - actual grid setup happens when
    // windows are created in the OS environment
    
    // Auto-initialize for any pre-existing windows
    setTimeout(() => {
        document.querySelectorAll('[id^="grid-container1-canvas-"]').forEach(canvas => {
            const windowId = canvas.id.replace('grid-container1-canvas-', '');
            initializeQuantumGrid(windowId);
        });
    }, 1000);
});

// MutationObserver to detect when a new encryption window is opened
// and initialize grid for it
const gridObserver = new MutationObserver(mutations => {
    mutations.forEach(mutation => {
        if (mutation.type === 'childList' && mutation.addedNodes.length > 0) {
            mutation.addedNodes.forEach(node => {
                if (node.nodeType === Node.ELEMENT_NODE) {
                    // Check if this is a new window containing grid canvases
                    const canvases = node.querySelectorAll('[id^="grid-container1-canvas-"]');
                    canvases.forEach(canvas => {
                        const windowId = canvas.id.replace('grid-container1-canvas-', '');
                        // Initialize grid for this new window
                        setTimeout(() => {
                            initializeQuantumGrid(windowId);
                        }, 500); // Small delay to ensure all elements are loaded
                    });
                }
            });
        }
    });
});

// Start observing the document for added nodes
document.addEventListener('DOMContentLoaded', function() {
    gridObserver.observe(document.body, { childList: true, subtree: true });
});