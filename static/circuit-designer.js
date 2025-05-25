// Quantum Circuit Designer for QuantoniumOS
// Integrates with the quantum-matrix.js visualization

class QuantumCircuitDesigner {
    constructor(containerId, visualizer) {
        this.container = document.getElementById(containerId);
        this.visualizer = visualizer; // Reference to QuantumMatrixVisualizer
        this.gates = []; // Array of gates applied to the circuit
        this.qubitCount = this.visualizer ? this.visualizer.qubitCount : 8;
        
        // Initialize the designer
        this.init();
    }
    
    init() {
        if (!this.container) return;
        
        // Create the main design elements
        this.createDesignerLayout();
        this.populateGatePalette();
        this.createCircuitGrid();
        this.setupEventListeners();
    }
    
    createDesignerLayout() {
        this.container.innerHTML = `
            <div class="circuit-designer-container">
                <div class="designer-header">
                    <h3>Quantum Circuit Designer</h3>
                    <div class="designer-controls">
                        <button id="run-circuit" class="designer-button">Run Circuit</button>
                        <button id="reset-circuit" class="designer-button">Reset Circuit</button>
                        <button id="hide-designer" class="designer-button">Hide Designer</button>
                    </div>
                </div>
                <div class="designer-content">
                    <div class="gate-palette" id="gate-palette">
                        <h4>Gate Palette</h4>
                        <div class="gate-list" id="gate-list"></div>
                    </div>
                    <div class="circuit-grid-container">
                        <h4>Circuit Design</h4>
                        <div class="qubit-labels" id="qubit-labels"></div>
                        <div class="circuit-grid" id="circuit-grid"></div>
                    </div>
                </div>
                <div class="circuit-output">
                    <h4>Circuit Information</h4>
                    <div id="circuit-info" class="circuit-info"></div>
                </div>
            </div>
        `;
        
        // Add CSS styles for the designer
        this.addDesignerStyles();
    }
    
    addDesignerStyles() {
        const style = document.createElement('style');
        style.id = 'circuit-designer-styles';
        style.textContent = `
            .circuit-designer-container {
                background-color: #111;
                border-radius: 5px;
                padding: 15px;
                margin-bottom: 20px;
                color: white;
                font-family: Arial, sans-serif;
            }
            
            .designer-header {
                display: flex;
                justify-content: space-between;
                align-items: center;
                margin-bottom: 15px;
                border-bottom: 1px solid #333;
                padding-bottom: 10px;
            }
            
            .designer-header h3 {
                margin: 0;
                color: #00b7ff;
            }
            
            .designer-controls {
                display: flex;
                gap: 10px;
            }
            
            .designer-button {
                background-color: #00475e;
                color: white;
                border: none;
                padding: 8px 12px;
                border-radius: 4px;
                cursor: pointer;
                font-size: 14px;
                transition: background-color 0.3s;
            }
            
            .designer-button:hover {
                background-color: #00b7ff;
            }
            
            .designer-content {
                display: flex;
                gap: 20px;
                margin-bottom: 15px;
            }
            
            .gate-palette {
                flex: 0 0 150px;
                background-color: #0a0a0a;
                border-radius: 4px;
                padding: 10px;
            }
            
            .gate-palette h4 {
                margin-top: 0;
                margin-bottom: 10px;
                font-size: 16px;
                color: #00b7ff;
                text-align: center;
            }
            
            .gate-list {
                display: flex;
                flex-direction: column;
                gap: 8px;
            }
            
            .gate-item {
                background-color: #1a1a1a;
                border: 1px solid #333;
                border-radius: 4px;
                padding: 8px;
                text-align: center;
                cursor: grab;
                transition: background-color 0.2s;
            }
            
            .gate-item:hover {
                background-color: #222;
                border-color: #00b7ff;
            }
            
            .gate-item.selected {
                background-color: #00475e;
                border-color: #00b7ff;
            }
            
            .circuit-grid-container {
                flex: 1;
                background-color: #0a0a0a;
                border-radius: 4px;
                padding: 10px;
                overflow-x: auto;
            }
            
            .circuit-grid-container h4 {
                margin-top: 0;
                margin-bottom: 10px;
                font-size: 16px;
                color: #00b7ff;
                text-align: center;
            }
            
            .qubit-labels {
                display: flex;
                flex-direction: column;
                gap: 10px;
                float: left;
                margin-right: 10px;
                margin-top: 30px;
            }
            
            .qubit-label {
                height: 30px;
                display: flex;
                align-items: center;
                padding: 0 10px;
                background-color: #1a1a1a;
                border-radius: 4px;
                font-family: monospace;
            }
            
            .circuit-grid {
                display: flex;
                flex-direction: column;
                gap: 10px;
                overflow-x: auto;
                min-height: 200px;
            }
            
            .circuit-row {
                display: flex;
                height: 30px;
                position: relative;
            }
            
            .circuit-wire {
                position: absolute;
                left: 0;
                right: 0;
                top: 50%;
                height: 2px;
                background-color: rgba(0, 183, 255, 0.5);
                z-index: 1;
            }
            
            .circuit-cell {
                width: 40px;
                height: 30px;
                border: 1px dashed #333;
                display: flex;
                justify-content: center;
                align-items: center;
                position: relative;
                z-index: 2;
                background-color: transparent;
                transition: background-color 0.2s;
            }
            
            .circuit-cell:hover {
                background-color: rgba(0, 183, 255, 0.1);
                border-style: solid;
            }
            
            .circuit-cell.gate {
                background-color: #00475e;
                border: 1px solid #00b7ff;
                cursor: pointer;
            }
            
            .circuit-cell.control {
                position: relative;
            }
            
            .circuit-cell.control::before {
                content: '';
                position: absolute;
                width: 10px;
                height: 10px;
                background-color: #00b7ff;
                border-radius: 50%;
            }
            
            .circuit-output {
                background-color: #0a0a0a;
                border-radius: 4px;
                padding: 10px;
            }
            
            .circuit-output h4 {
                margin-top: 0;
                margin-bottom: 10px;
                font-size: 16px;
                color: #00b7ff;
                text-align: center;
            }
            
            .circuit-info {
                font-family: monospace;
                white-space: pre-wrap;
                background-color: #1a1a1a;
                padding: 10px;
                border-radius: 4px;
                max-height: 150px;
                overflow-y: auto;
            }
            
            .connection-line {
                position: absolute;
                background-color: #00b7ff;
                width: 2px;
                z-index: 1;
            }
        `;
        
        document.head.appendChild(style);
    }
    
    populateGatePalette() {
        const gateList = document.getElementById('gate-list');
        if (!gateList) return;
        
        // Add standard quantum gates
        const gates = [
            { id: 'h', name: 'H', description: 'Hadamard' },
            { id: 'x', name: 'X', description: 'Pauli X (NOT)' },
            { id: 'y', name: 'Y', description: 'Pauli Y' },
            { id: 'z', name: 'Z', description: 'Pauli Z' },
            { id: 's', name: 'S', description: 'Phase' },
            { id: 't', name: 'T', description: 'T Gate' },
            { id: 'cx', name: 'CX', description: 'Controlled X (CNOT)' },
            { id: 'cz', name: 'CZ', description: 'Controlled Z' },
            { id: 'swap', name: 'SWAP', description: 'Swap Qubits' }
        ];
        
        gates.forEach(gate => {
            const gateItem = document.createElement('div');
            gateItem.className = 'gate-item';
            gateItem.dataset.gate = gate.id;
            gateItem.title = gate.description;
            gateItem.textContent = gate.name;
            gateItem.draggable = true;
            
            gateItem.addEventListener('dragstart', (e) => {
                e.dataTransfer.setData('text/plain', gate.id);
                e.dataTransfer.effectAllowed = 'copy';
            });
            
            gateList.appendChild(gateItem);
        });
    }
    
    createCircuitGrid() {
        // Create qubit labels
        const qubitLabels = document.getElementById('qubit-labels');
        if (!qubitLabels) return;
        
        // Default grid size
        const columns = 10;
        const visibleQubits = Math.min(this.qubitCount, 8); // Limit visible qubits for UI
        
        // Create qubit labels
        for (let i = 0; i < visibleQubits; i++) {
            const label = document.createElement('div');
            label.className = 'qubit-label';
            label.textContent = `q${i}`;
            qubitLabels.appendChild(label);
        }
        
        // Create circuit grid
        const circuitGrid = document.getElementById('circuit-grid');
        if (!circuitGrid) return;
        
        for (let i = 0; i < visibleQubits; i++) {
            const row = document.createElement('div');
            row.className = 'circuit-row';
            row.dataset.qubit = i;
            
            // Add circuit wire
            const wire = document.createElement('div');
            wire.className = 'circuit-wire';
            row.appendChild(wire);
            
            // Add cells for gates
            for (let j = 0; j < columns; j++) {
                const cell = document.createElement('div');
                cell.className = 'circuit-cell';
                cell.dataset.qubit = i;
                cell.dataset.step = j;
                
                // Make cell a drop target for gates
                cell.addEventListener('dragover', (e) => {
                    e.preventDefault();
                    e.dataTransfer.dropEffect = 'copy';
                });
                
                cell.addEventListener('drop', (e) => {
                    e.preventDefault();
                    const gateType = e.dataTransfer.getData('text/plain');
                    this.addGateToCircuit(gateType, i, j);
                });
                
                row.appendChild(cell);
            }
            
            circuitGrid.appendChild(row);
        }
    }
    
    setupEventListeners() {
        // Run circuit button
        const runButton = document.getElementById('run-circuit');
        if (runButton) {
            runButton.addEventListener('click', () => {
                this.runCircuit();
            });
        }
        
        // Reset circuit button
        const resetButton = document.getElementById('reset-circuit');
        if (resetButton) {
            resetButton.addEventListener('click', () => {
                this.resetCircuit();
            });
        }
        
        // Hide designer button
        const hideButton = document.getElementById('hide-designer');
        if (hideButton) {
            hideButton.addEventListener('click', () => {
                this.hideDesigner();
            });
        }
    }
    
    addGateToCircuit(gateType, qubit, step) {
        // Find the cell
        const cell = document.querySelector(`.circuit-cell[data-qubit="${qubit}"][data-step="${step}"]`);
        if (!cell) return;
        
        // Clear existing gate if any
        cell.innerHTML = '';
        cell.className = 'circuit-cell';
        
        // Create new gate
        const gate = { type: gateType, target: qubit, step: step };
        
        // Check if it's a two-qubit gate
        if (gateType === 'cx' || gateType === 'cz' || gateType === 'swap') {
            // Mark as control point, will need a second click to complete
            cell.className = 'circuit-cell control';
            cell.textContent = '•';
            
            // Store temporarily as the first part of a two-qubit gate
            this.pendingControlGate = {
                type: gateType,
                control: qubit,
                controlStep: step
            };
            
            // Update circuit info
            this.updateCircuitInfo('Select target qubit for ' + gateType.toUpperCase() + ' gate');
            
            // Add event listeners to all other cells for selecting the target
            document.querySelectorAll('.circuit-cell').forEach(otherCell => {
                if (otherCell !== cell) {
                    const targetQubit = parseInt(otherCell.dataset.qubit);
                    const targetStep = parseInt(otherCell.dataset.step);
                    
                    // Only allow cells in the same column (time step)
                    if (targetStep === step && targetQubit !== qubit) {
                        otherCell.addEventListener('click', this.handleTargetSelection.bind(this, targetQubit, targetStep), { once: true });
                        otherCell.classList.add('potential-target');
                    }
                }
            });
        } else {
            // Single-qubit gate
            cell.className = 'circuit-cell gate';
            cell.textContent = gateType.toUpperCase();
            
            // Add to circuit
            this.gates.push(gate);
            
            // Update circuit info
            this.updateCircuitInfo();
        }
    }
    
    handleTargetSelection(targetQubit, targetStep) {
        if (!this.pendingControlGate) return;
        
        // Complete the two-qubit gate
        const gate = {
            type: this.pendingControlGate.type,
            control: this.pendingControlGate.control,
            target: targetQubit,
            step: targetStep
        };
        
        // Add to circuit
        this.gates.push(gate);
        
        // Update target cell
        const targetCell = document.querySelector(`.circuit-cell[data-qubit="${targetQubit}"][data-step="${targetStep}"]`);
        if (targetCell) {
            targetCell.className = 'circuit-cell gate';
            targetCell.textContent = gate.type.toUpperCase();
        }
        
        // Draw connection line
        this.drawConnectionLine(gate.control, gate.target, gate.step);
        
        // Reset pending gate
        this.pendingControlGate = null;
        
        // Remove potential-target class from all cells
        document.querySelectorAll('.potential-target').forEach(cell => {
            cell.classList.remove('potential-target');
        });
        
        // Update circuit info
        this.updateCircuitInfo();
    }
    
    drawConnectionLine(controlQubit, targetQubit, step) {
        const circuitGrid = document.getElementById('circuit-grid');
        if (!circuitGrid) return;
        
        const line = document.createElement('div');
        line.className = 'connection-line';
        
        // Get the control and target cells
        const controlCell = document.querySelector(`.circuit-cell[data-qubit="${controlQubit}"][data-step="${step}"]`);
        const targetCell = document.querySelector(`.circuit-cell[data-qubit="${targetQubit}"][data-step="${step}"]`);
        
        if (!controlCell || !targetCell) return;
        
        // Get cell positions
        const controlRect = controlCell.getBoundingClientRect();
        const targetRect = targetCell.getBoundingClientRect();
        const gridRect = circuitGrid.getBoundingClientRect();
        
        // Calculate relative positions
        const left = controlRect.left - gridRect.left + controlRect.width / 2;
        const top = Math.min(controlRect.top, targetRect.top) - gridRect.top + controlCell.offsetHeight / 2;
        const height = Math.abs(targetRect.top - controlRect.top);
        
        // Position the line
        line.style.left = `${left}px`;
        line.style.top = `${top}px`;
        line.style.height = `${height}px`;
        
        // Add to grid
        circuitGrid.appendChild(line);
    }
    
    updateCircuitInfo(message) {
        const infoElement = document.getElementById('circuit-info');
        if (!infoElement) return;
        
        if (message) {
            infoElement.textContent = message;
            return;
        }
        
        // Format gates for display
        let info = 'Circuit contains ' + this.gates.length + ' gates:\n';
        
        this.gates.forEach((gate, index) => {
            if (gate.control !== undefined) {
                info += `${index + 1}. ${gate.type.toUpperCase()} gate: control=q${gate.control}, target=q${gate.target}, step=${gate.step}\n`;
            } else {
                info += `${index + 1}. ${gate.type.toUpperCase()} gate: target=q${gate.target}, step=${gate.step}\n`;
            }
        });
        
        infoElement.textContent = info;
    }
    
    runCircuit() {
        if (!this.visualizer) {
            this.updateCircuitInfo('Error: Quantum visualizer not available');
            return;
        }
        
        // Sort gates by step to apply them in the correct order
        const sortedGates = [...this.gates].sort((a, b) => a.step - b.step);
        
        // Reset the visualizer
        this.visualizer.generateQuantumState();
        
        // Apply each gate
        let success = true;
        sortedGates.forEach(gate => {
            const result = this.visualizer.applyGate({
                type: gate.type,
                target: gate.target,
                control: gate.control
            });
            
            if (!result) {
                success = false;
                this.updateCircuitInfo(`Error applying gate: ${gate.type} at step ${gate.step}`);
            }
        });
        
        if (success) {
            this.updateCircuitInfo('Circuit executed successfully! Check visualization for results.');
        }
    }
    
    resetCircuit() {
        // Clear the circuit grid
        document.querySelectorAll('.circuit-cell').forEach(cell => {
            cell.innerHTML = '';
            cell.className = 'circuit-cell';
        });
        
        // Remove connection lines
        document.querySelectorAll('.connection-line').forEach(line => {
            line.remove();
        });
        
        // Reset gates
        this.gates = [];
        this.pendingControlGate = null;
        
        // Reset visualizer
        if (this.visualizer) {
            this.visualizer.generateQuantumState();
        }
        
        // Update info
        this.updateCircuitInfo('Circuit reset');
    }
    
    hideDesigner() {
        // Toggle visibility
        this.container.style.display = this.container.style.display === 'none' ? 'block' : 'none';
        
        // Change button text
        const hideButton = document.getElementById('hide-designer');
        if (hideButton) {
            hideButton.textContent = this.container.style.display === 'none' ? 'Show Designer' : 'Hide Designer';
        }
    }
    
    // Update the designer to match current visualizer state
    updateDesigner() {
        if (this.visualizer) {
            this.qubitCount = this.visualizer.qubitCount;
            // If needed, recreate the circuit grid with the new qubit count
        }
    }
}

// Allow the designer to be initialized from outside
window.initQuantumCircuitDesigner = function(containerId, visualizerId) {
    // Wait for DOM to be ready
    document.addEventListener('DOMContentLoaded', () => {
        // Find the visualizer instance
        const visualizer = window.quantumVisualizer;
        
        // Create the designer
        window.circuitDesigner = new QuantumCircuitDesigner(containerId, visualizer);
    });
};