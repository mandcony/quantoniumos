// QuantoniumOS Quantum Matrix Visualization
// High-performance visualization for 512-qubit systems

// Main visualization controller
class QuantumMatrixVisualizer {
    constructor(canvasId) {
        this.canvas = document.getElementById(canvasId);
        this.ctx = this.canvas.getContext('2d');
        this.qubitCount = 64; // Default
        this.maxQubits = 512; // Maximum supported
        this.zoomLevel = 1;
        this.viewportX = 0;
        this.viewportY = 0;
        this.isDragging = false;
        this.mode = 'matrix'; // Default visualization mode
        
        // Quantum state data
        this.eigenvalues = [];
        this.eigenvectors = [];
        this.entanglementStrength = 0.5;
        
        // Initialize event handlers
        this.initEventHandlers();
        
        // Generate initial quantum state
        this.generateQuantumState();
        
        // Start rendering
        this.draw();
    }
    
    // Set up event handlers for interaction
    initEventHandlers() {
        // Mouse events for panning
        this.canvas.addEventListener('mousedown', (e) => {
            this.isDragging = true;
            this.lastMouseX = e.clientX;
            this.lastMouseY = e.clientY;
            this.canvas.style.cursor = 'grabbing';
        });
        
        document.addEventListener('mousemove', (e) => {
            if (!this.isDragging) return;
            
            const deltaX = e.clientX - this.lastMouseX;
            const deltaY = e.clientY - this.lastMouseY;
            
            this.viewportX += deltaX / this.zoomLevel;
            this.viewportY += deltaY / this.zoomLevel;
            
            this.lastMouseX = e.clientX;
            this.lastMouseY = e.clientY;
            
            this.draw();
        });
        
        document.addEventListener('mouseup', () => {
            this.isDragging = false;
            this.canvas.style.cursor = 'grab';
        });
        
        // Wheel events for zooming
        this.canvas.addEventListener('wheel', (e) => {
            e.preventDefault();
            
            const scaleFactor = e.deltaY > 0 ? 0.9 : 1.1;
            this.zoomLevel *= scaleFactor;
            
            // Limit zoom level
            this.zoomLevel = Math.max(0.1, Math.min(10, this.zoomLevel));
            
            this.draw();
        });
        
        // Window resize handler
        window.addEventListener('resize', () => {
            this.resizeCanvas();
            this.draw();
        });
    }
    
    // Resize canvas to match container
    resizeCanvas() {
        const container = this.canvas.parentElement;
        this.canvas.width = container.clientWidth;
        this.canvas.height = container.clientHeight;
    }
    
    // Update qubit count
    setQubitCount(count) {
        this.qubitCount = Math.min(Math.max(2, count), this.maxQubits);
        this.generateQuantumState();
        this.draw();
    }
    
    // Set visualization mode
    setMode(mode) {
        this.mode = mode;
        this.draw();
    }
    
    // Set entanglement strength
    setEntanglementStrength(value) {
        this.entanglementStrength = Math.min(Math.max(0, value), 1);
        this.draw();
    }
    
    // Apply quantum gate to the state
    applyGate(gate) {
        // Validate gate parameters
        if (gate.target >= this.qubitCount || gate.target < 0) {
            return false;
        }
        
        if ((gate.type === 'cx' || gate.type === 'cz' || gate.type === 'swap') && 
            (gate.control >= this.qubitCount || gate.control < 0 || gate.control === gate.target)) {
            return false;
        }
        
        // Apply gate effect to eigenvalues/eigenvectors
        switch (gate.type) {
            case 'h': // Hadamard gate
                if (gate.target < this.eigenvalues.length) {
                    // Rotate the eigenvalue by 45 degrees
                    const ev = this.eigenvalues[gate.target];
                    const real = ev.real;
                    const imag = ev.imag;
                    const angle = Math.PI / 4; // 45 degrees
                    
                    this.eigenvalues[gate.target] = {
                        real: real * Math.cos(angle) - imag * Math.sin(angle),
                        imag: real * Math.sin(angle) + imag * Math.cos(angle)
                    };
                }
                break;
                
            case 'x': // Pauli X (NOT gate)
                if (gate.target < this.eigenvalues.length) {
                    // Flip the eigenvalue
                    this.eigenvalues[gate.target] = {
                        real: -this.eigenvalues[gate.target].real,
                        imag: -this.eigenvalues[gate.target].imag
                    };
                }
                break;
                
            case 'cx': // CNOT gate
                if (gate.target < this.eigenvalues.length && gate.control < this.eigenvalues.length) {
                    // Entangle the target with the control
                    // We'll create correlation in the eigenvectors
                    for (let i = 0; i < this.eigenvectors.length; i++) {
                        if (this.eigenvectors[i][gate.control].real > 0) {
                            this.eigenvectors[i][gate.target] = {
                                real: -this.eigenvectors[i][gate.target].real,
                                imag: -this.eigenvectors[i][gate.target].imag
                            };
                        }
                    }
                }
                break;
                
            // Implement other gates similarly
        }
        
        this.draw();
        return true;
    }
    
    // Generate quantum state with eigenvalues and eigenvectors
    generateQuantumState() {
        // Generate eigenvalues for the quantum state (512 max)
        this.eigenvalues = [];
        this.eigenvectors = [];
        
        // Create pseudo-eigenvalues for visualization
        for (let i = 0; i < this.qubitCount; i++) {
            // Random complex eigenvalue
            this.eigenvalues.push({
                real: Math.random() * 2 - 1,
                imag: Math.random() * 2 - 1
            });
            
            // Create eigenvector for each eigenvalue
            const eigenvector = [];
            for (let j = 0; j < this.qubitCount; j++) {
                eigenvector.push({
                    real: Math.random() * 2 - 1,
                    imag: Math.random() * 2 - 1
                });
            }
            this.eigenvectors.push(eigenvector);
        }
    }
    
    // Main rendering function
    draw() {
        if (!this.ctx) return;
        
        // Clear canvas
        this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
        
        // Apply viewport transformations
        this.ctx.save();
        this.ctx.translate(this.canvas.width / 2 + this.viewportX, this.canvas.height / 2 + this.viewportY);
        this.ctx.scale(this.zoomLevel, this.zoomLevel);
        
        // Choose visualization method based on selected mode
        switch (this.mode) {
            case 'matrix':
                this.drawMatrixVisualization();
                break;
            case 'network':
                this.drawNetworkVisualization();
                break;
            case 'bloch':
                this.drawBlochSphereVisualization();
                break;
            case 'heatmap':
                this.drawHeatmapVisualization();
                break;
            default:
                this.drawMatrixVisualization();
        }
        
        this.ctx.restore();
        
        // Draw UI overlay
        this.drawUIOverlay();
    }
    
    // Matrix visualization - shows eigenvalues and eigenvectors as a matrix
    drawMatrixVisualization() {
        const cellSize = Math.max(4, 600 / Math.sqrt(this.qubitCount));
        const matrixSize = Math.ceil(Math.sqrt(this.qubitCount)) * cellSize;
        const startX = -matrixSize / 2;
        const startY = -matrixSize / 2;
        
        // Draw background grid
        this.ctx.strokeStyle = 'rgba(0, 183, 255, 0.2)';
        this.ctx.lineWidth = 0.5;
        
        // Draw eigenvalues along diagonal
        for (let i = 0; i < this.qubitCount && i < this.eigenvectors.length; i++) {
            const row = Math.floor(i / Math.sqrt(this.qubitCount));
            const col = i % Math.sqrt(this.qubitCount);
            const x = startX + col * cellSize;
            const y = startY + row * cellSize;
            
            // Draw cell
            const intensity = Math.sqrt(
                Math.pow(this.eigenvalues[i].real, 2) + 
                Math.pow(this.eigenvalues[i].imag, 2)
            ) / Math.sqrt(2);
            
            // Color based on phase angle
            const phase = Math.atan2(this.eigenvalues[i].imag, this.eigenvalues[i].real);
            const hue = ((phase + Math.PI) / (2 * Math.PI)) * 360;
            
            this.ctx.fillStyle = `hsla(${hue}, 100%, 50%, ${intensity * 0.7})`;
            this.ctx.fillRect(x, y, cellSize, cellSize);
            
            // Draw cell border
            this.ctx.strokeRect(x, y, cellSize, cellSize);
            
            // Draw eigenvalue indicator
            if (cellSize > 10) {
                // Only show details if cells are large enough
                this.ctx.fillStyle = 'rgba(255, 255, 255, 0.9)';
                this.ctx.font = `${Math.max(8, cellSize/5)}px monospace`;
                this.ctx.textAlign = 'center';
                this.ctx.textBaseline = 'middle';
                const valueDisplay = i.toString().padStart(2, '0');
                this.ctx.fillText(valueDisplay, x + cellSize/2, y + cellSize/2);
            }
        }
        
        // Draw label for the current view
        this.ctx.fillStyle = 'rgba(0, 183, 255, 0.8)';
        this.ctx.font = '16px Arial';
        this.ctx.textAlign = 'center';
        this.ctx.fillText(`${this.qubitCount}×${this.qubitCount} Matrix Eigenvector Visualization`, 0, -matrixSize/2 - 20);
    }
    
    // Network visualization - shows qubits as a network
    drawNetworkVisualization() {
        const radius = Math.min(this.canvas.width, this.canvas.height) * 0.4;
        
        // Draw background circle
        this.ctx.strokeStyle = 'rgba(0, 183, 255, 0.2)';
        this.ctx.lineWidth = 1;
        this.ctx.beginPath();
        this.ctx.arc(0, 0, radius, 0, Math.PI * 2);
        this.ctx.stroke();
        
        // Calculate node positions around a circle
        const nodePositions = [];
        for (let i = 0; i < this.qubitCount; i++) {
            const angle = (i / this.qubitCount) * Math.PI * 2;
            const x = Math.cos(angle) * radius;
            const y = Math.sin(angle) * radius;
            nodePositions.push({ x, y, id: i });
        }
        
        // Draw connections based on entanglement
        const maxConnections = Math.min(1000, this.qubitCount * this.qubitCount / 4); // Limit connections for performance
        const connectionsToShow = Math.floor(maxConnections * this.entanglementStrength);
        
        for (let i = 0; i < this.qubitCount; i++) {
            // Connect to other nodes based on entanglement
            for (let j = i + 1; j < this.qubitCount; j++) {
                // Only draw a subset of connections based on entanglement strength
                if (Math.random() > (1 - this.entanglementStrength) * 0.95) continue;
                
                // Calculate entanglement strength from eigenvectors
                let entanglementMeasure = 0;
                for (let k = 0; k < this.eigenvectors.length && k < 10; k++) { // Sample a few eigenvectors for performance
                    if (this.eigenvectors[k][i] && this.eigenvectors[k][j]) {
                        const vi = this.eigenvectors[k][i];
                        const vj = this.eigenvectors[k][j];
                        entanglementMeasure += Math.abs(vi.real * vj.real + vi.imag * vj.imag);
                    }
                }
                
                const normalizedMeasure = entanglementMeasure / Math.sqrt(10);
                
                // Only draw connections above a threshold
                if (normalizedMeasure > 0.1) {
                    const alpha = Math.min(1, normalizedMeasure * 0.5);
                    this.ctx.strokeStyle = `rgba(0, 183, 255, ${alpha})`;
                    this.ctx.lineWidth = Math.max(0.5, normalizedMeasure * 2);
                    
                    this.ctx.beginPath();
                    this.ctx.moveTo(nodePositions[i].x, nodePositions[i].y);
                    this.ctx.lineTo(nodePositions[j].x, nodePositions[j].y);
                    this.ctx.stroke();
                }
            }
        }
        
        // Draw nodes
        for (let i = 0; i < nodePositions.length; i++) {
            const { x, y, id } = nodePositions[i];
            
            // Base color on eigenvalue
            let nodeColor = 'rgba(0, 183, 255, 0.8)';
            const nodeSize = 8;
            
            if (i < this.eigenvalues.length) {
                // Color based on phase angle of eigenvalue
                const phase = Math.atan2(this.eigenvalues[i].imag, this.eigenvalues[i].real);
                const hue = ((phase + Math.PI) / (2 * Math.PI)) * 360;
                nodeColor = `hsla(${hue}, 100%, 50%, 0.8)`;
            }
            
            // Draw node
            this.ctx.fillStyle = nodeColor;
            this.ctx.beginPath();
            this.ctx.arc(x, y, nodeSize, 0, Math.PI * 2);
            this.ctx.fill();
            
            // Draw node label if zoomed in enough
            if (this.zoomLevel > 0.8 && id % Math.ceil(this.qubitCount / 100) === 0) {
                this.ctx.fillStyle = 'white';
                this.ctx.font = '10px monospace';
                this.ctx.textAlign = 'center';
                this.ctx.textBaseline = 'middle';
                this.ctx.fillText(id.toString(), x, y);
            }
        }
        
        // Draw label
        this.ctx.fillStyle = 'rgba(0, 183, 255, 0.8)';
        this.ctx.font = '16px Arial';
        this.ctx.textAlign = 'center';
        this.ctx.fillText(`${this.qubitCount}-Qubit Network Visualization`, 0, -radius - 20);
    }
    
    // Draw Bloch sphere representation (for fewer qubits)
    drawBlochSphereVisualization() {
        const maxVisibleQubits = Math.min(this.qubitCount, 36); // Limit to make it readable
        const gridSize = Math.ceil(Math.sqrt(maxVisibleQubits));
        
        // Calculate cell size based on available space
        const cellSize = Math.min(this.canvas.width, this.canvas.height) * 0.8 / gridSize;
        const startX = -(gridSize * cellSize) / 2;
        const startY = -(gridSize * cellSize) / 2;
        
        for (let i = 0; i < maxVisibleQubits; i++) {
            const row = Math.floor(i / gridSize);
            const col = i % gridSize;
            const x = startX + col * cellSize + cellSize / 2;
            const y = startY + row * cellSize + cellSize / 2;
            
            // Draw Bloch sphere
            this.drawBlochSphere(x, y, cellSize * 0.4, i);
        }
        
        // Draw label with qubit count info
        this.ctx.fillStyle = 'rgba(0, 183, 255, 0.8)';
        this.ctx.font = '16px Arial';
        this.ctx.textAlign = 'center';
        this.ctx.fillText(`Bloch Sphere Visualization (showing ${maxVisibleQubits} of ${this.qubitCount} qubits)`, 0, startY - 20);
    }
    
    // Draw heatmap visualization
    drawHeatmapVisualization() {
        // For large qubit counts, we need an efficient heatmap
        const maxDisplaySize = 200; // Maximum size that works well in browser
        const cellSize = Math.max(2, maxDisplaySize / Math.sqrt(this.qubitCount));
        const visibleRows = Math.min(this.qubitCount, Math.ceil(maxDisplaySize / cellSize));
        const visibleCols = Math.min(this.qubitCount, Math.ceil(maxDisplaySize / cellSize));
        
        // Create an offscreen buffer for performance
        const buffer = new Uint8ClampedArray(visibleRows * visibleCols * 4);
        
        // Generate heatmap values
        for (let i = 0; i < visibleRows; i++) {
            for (let j = 0; j < visibleCols; j++) {
                let intensity = 0;
                
                // Calculate interaction strength between qubits i and j
                if (i < this.eigenvectors.length && j < this.eigenvectors[0].length) {
                    // Sample a subset of eigenvectors for performance
                    const sampleSize = Math.min(10, this.eigenvectors.length);
                    for (let k = 0; k < sampleSize; k++) {
                        if (this.eigenvectors[k] && this.eigenvectors[k][i] && this.eigenvectors[k][j]) {
                            const vi = this.eigenvectors[k][i];
                            const vj = this.eigenvectors[k][j];
                            intensity += Math.abs(vi.real * vj.real + vi.imag * vj.imag);
                        }
                    }
                    intensity /= sampleSize; // Normalize
                }
                
                // Set pixel color in buffer (RGBA)
                const index = (i * visibleCols + j) * 4;
                buffer[index] = 0; // R
                buffer[index + 1] = 183 * intensity; // G
                buffer[index + 2] = 255 * intensity; // B
                buffer[index + 3] = 255 * Math.min(0.9, intensity * 3); // A
            }
        }
        
        // Create ImageData and put it on canvas
        const imageData = new ImageData(buffer, visibleCols, visibleRows);
        const offscreenCanvas = document.createElement('canvas');
        offscreenCanvas.width = visibleCols;
        offscreenCanvas.height = visibleRows;
        const offscreenCtx = offscreenCanvas.getContext('2d');
        offscreenCtx.putImageData(imageData, 0, 0);
        
        // Draw heatmap image
        const drawWidth = visibleCols * cellSize;
        const drawHeight = visibleRows * cellSize;
        this.ctx.drawImage(
            offscreenCanvas, 
            -drawWidth / 2, 
            -drawHeight / 2, 
            drawWidth, 
            drawHeight
        );
        
        // Draw visualization label
        this.ctx.fillStyle = 'rgba(0, 183, 255, 0.8)';
        this.ctx.font = '16px Arial';
        this.ctx.textAlign = 'center';
        this.ctx.fillText(`${this.qubitCount}×${this.qubitCount} Quantum State Heatmap`, 0, -drawHeight / 2 - 20);
    }
    
    // Helper function to draw a Bloch sphere for a single qubit
    drawBlochSphere(centerX, centerY, radius, qubitIndex) {
        // Draw sphere
        this.ctx.strokeStyle = 'rgba(0, 183, 255, 0.4)';
        this.ctx.lineWidth = 1;
        
        // Draw circle (sphere projection)
        this.ctx.beginPath();
        this.ctx.arc(centerX, centerY, radius, 0, Math.PI * 2);
        this.ctx.stroke();
        
        // Draw axes
        this.ctx.beginPath();
        this.ctx.moveTo(centerX - radius, centerY);
        this.ctx.lineTo(centerX + radius, centerY);
        this.ctx.moveTo(centerX, centerY - radius);
        this.ctx.lineTo(centerX, centerY + radius);
        this.ctx.stroke();
        
        // Get qubit state from eigenvalues/vectors
        if (qubitIndex < this.eigenvalues.length) {
            // Calculate direction from eigenvalue
            const ev = this.eigenvalues[qubitIndex];
            const theta = Math.atan2(ev.imag, ev.real) * 2; // Angle in XY plane
            const phi = Math.PI / 2 * (1 - Math.sqrt(ev.real*ev.real + ev.imag*ev.imag)); // Angle from Z axis
            
            // Convert spherical to cartesian
            const x = radius * Math.sin(phi) * Math.cos(theta);
            const y = radius * Math.sin(phi) * Math.sin(theta);
            
            // Draw state vector
            this.ctx.strokeStyle = 'rgba(0, 183, 255, 0.9)';
            this.ctx.lineWidth = 2;
            this.ctx.beginPath();
            this.ctx.moveTo(centerX, centerY);
            this.ctx.lineTo(centerX + x, centerY + y);
            this.ctx.stroke();
            
            // Draw qubit label
            this.ctx.fillStyle = 'white';
            this.ctx.font = '10px monospace';
            this.ctx.textAlign = 'center';
            this.ctx.textBaseline = 'middle';
            this.ctx.fillText(qubitIndex.toString(), centerX, centerY - radius - 10);
        }
    }
    
    // Draw UI overlay with information and stats
    drawUIOverlay() {
        // Draw info text
        this.ctx.fillStyle = 'rgba(255, 255, 255, 0.8)';
        this.ctx.font = '12px monospace';
        this.ctx.textAlign = 'left';
        this.ctx.textBaseline = 'top';
        
        const infoText = [
            `Qubits: ${this.qubitCount}/${this.maxQubits}`,
            `Mode: ${this.mode}`,
            `Zoom: ${this.zoomLevel.toFixed(1)}x`,
            `Entanglement: ${this.entanglementStrength.toFixed(1)}`
        ];
        
        infoText.forEach((text, i) => {
            this.ctx.fillText(text, 10, 10 + i * 18);
        });
    }
    
    // Get state representation as text
    getStateText() {
        // For large qubit states, just show a summary
        if (this.qubitCount > 16) {
            return `Quantum state: ${this.qubitCount} qubits\nLarge quantum state - use visualization to explore`;
        }
        
        // For smaller states, show detailed representation
        let stateText = `Quantum state (${this.qubitCount} qubits):\n`;
        
        // Show first few eigenvalues
        stateText += "Eigenvalues:\n";
        const showCount = Math.min(8, this.eigenvalues.length);
        
        for (let i = 0; i < showCount; i++) {
            const ev = this.eigenvalues[i];
            stateText += `λ${i}: ${ev.real.toFixed(2)} + ${ev.imag.toFixed(2)}i\n`;
        }
        
        if (this.eigenvalues.length > showCount) {
            stateText += `... and ${this.eigenvalues.length - showCount} more\n`;
        }
        
        return stateText;
    }
}

// Initialize when the page loads
window.addEventListener('DOMContentLoaded', () => {
    console.log('Initializing Quantum Matrix Visualizer for 512-qubit system');
});