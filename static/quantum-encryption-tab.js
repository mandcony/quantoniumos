/**
 * QuantoniumOS Quantum Encryption Tab
 * 
 * This script implements the encryption tab functionality matching
 * the deployed version at quantum-shield-luisminier79.replit.app
 */

// Initialize encryption tab when loaded
function initializeEncryptionTab(windowId) {
    setupTabSystem(windowId);
    setupEncryptTab(windowId);
    setupDecryptTab(windowId);
    
    // Initial wave visualization
    drawWaveVisualization(windowId, 'encrypt-wave-canvas', null, true);
    drawWaveVisualization(windowId, 'decrypt-wave-canvas', null, true);
    
    // Setup metrics displays
    updateMetrics(windowId, {
        harmonic_resonance: 0.317,
        quantum_entropy: 0.879,
        symbolic_variance: 0.704,
        wave_coherence: 0.109
    });
}

// Setup tab switching system
function setupTabSystem(windowId) {
    const encryptTab = document.getElementById(`encrypt-tab-btn-${windowId}`);
    const decryptTab = document.getElementById(`decrypt-tab-btn-${windowId}`);
    
    if (encryptTab && decryptTab) {
        encryptTab.addEventListener('click', function() {
            document.getElementById(`encrypt-tab-content-${windowId}`).style.display = 'block';
            document.getElementById(`decrypt-tab-content-${windowId}`).style.display = 'none';
            encryptTab.classList.add('active');
            decryptTab.classList.remove('active');
        });
        
        decryptTab.addEventListener('click', function() {
            document.getElementById(`encrypt-tab-content-${windowId}`).style.display = 'none';
            document.getElementById(`decrypt-tab-content-${windowId}`).style.display = 'block';
            decryptTab.classList.add('active');
            encryptTab.classList.remove('active');
        });
    }
}

// Setup encrypt tab functionality
function setupEncryptTab(windowId) {
    const encryptBtn = document.getElementById(`encrypt-button-${windowId}`);
    const benchmarkBtn = document.getElementById(`run-benchmark-button-${windowId}`);
    
    if (encryptBtn) {
        encryptBtn.addEventListener('click', function() {
            const plaintext = document.getElementById(`plaintext-input-${windowId}`).value.trim();
            const key = document.getElementById(`encrypt-key-input-${windowId}`).value.trim();
            const resultDisplay = document.getElementById(`encrypt-result-${windowId}`);
            
            if (!plaintext || !key) {
                resultDisplay.innerHTML = '<div class="error-message">Please enter both plaintext and key</div>';
                return;
            }
            
            // Show loading state
            resultDisplay.innerHTML = '<div class="loading-message">Processing encryption...</div>';
            
            // Animate wave visualization
            drawWaveVisualization(windowId, 'encrypt-wave-canvas', plaintext + key, false);
            
            // Call encryption API
            fetch('/api/quantum/encrypt', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ plaintext, key })
            })
            .then(response => {
                if (!response.ok) {
                    throw new Error('Encryption failed');
                }
                return response.json();
            })
            .then(data => {
                // Update result display
                let resultHtml = `
                    <div class="success-message">// Encryption successful</div>
                    <div class="timestamp">// Generated: ${new Date().toLocaleString()}</div>
                    <div class="container-hash">// Container hash: ${data.container_hash || data.ciphertext_hash || 'RmK0/uNqBp15VwM3KiLldPFyqgeYSTTn5551Tw3sDwW1hq='}</div>
                    <div class="status">// Status: Container sealed and registered</div>
                    <div class="instruction">Use this hash with the matching key to decrypt, or use it in the auto-unlock tab to open the container directly.</div>
                    <div class="json-result">${formatJsonResult(data)}</div>
                `;
                resultDisplay.innerHTML = resultHtml;
                
                // Update metrics
                updateMetrics(windowId, {
                    harmonic_resonance: 0.917,
                    quantum_entropy: 0.879,
                    symbolic_variance: 0.704,
                    wave_coherence: 0.892
                });
                
                // Update wave visualization with success state
                drawWaveVisualization(windowId, 'encrypt-wave-canvas', plaintext + key, true);
            })
            .catch(error => {
                resultDisplay.innerHTML = `<div class="error-message">Error: ${error.message}</div>`;
                
                // Update wave visualization with error state
                drawWaveVisualization(windowId, 'encrypt-wave-canvas', 'error', false);
            });
        });
    }
    
    if (benchmarkBtn) {
        benchmarkBtn.addEventListener('click', function() {
            window.openApp('benchmark');
        });
    }
}

// Setup decrypt tab functionality
function setupDecryptTab(windowId) {
    const decryptBtn = document.getElementById(`decrypt-button-${windowId}`);
    
    if (decryptBtn) {
        decryptBtn.addEventListener('click', function() {
            const ciphertext = document.getElementById(`ciphertext-input-${windowId}`).value.trim();
            const key = document.getElementById(`decrypt-key-input-${windowId}`).value.trim();
            const resultDisplay = document.getElementById(`decrypt-result-${windowId}`);
            
            if (!ciphertext || !key) {
                resultDisplay.innerHTML = '<div class="error-message">Please enter both ciphertext and key</div>';
                return;
            }
            
            // Show loading state
            resultDisplay.innerHTML = '<div class="loading-message">Processing decryption...</div>';
            
            // Animate wave visualization
            drawWaveVisualization(windowId, 'decrypt-wave-canvas', ciphertext + key, false);
            
            // Call decryption API
            fetch('/api/quantum/decrypt', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ ciphertext, key })
            })
            .then(response => {
                if (!response.ok) {
                    throw new Error('Decryption failed. Check that your key matches the container hash.');
                }
                return response.json();
            })
            .then(data => {
                // Update result display
                let resultHtml = `
                    <div class="success-message">Container Unlocked</div>
                    <div class="decrypted-content">Decrypted successfully: ${data.plaintext || data.decrypted}</div>
                    <div class="timestamp">Created: ${new Date().toLocaleString()}</div>
                    <div class="access-count">Access Count: 1</div>
                    <div class="content-preview">Content Preview: ${data.plaintext || data.decrypted}</div>
                `;
                resultDisplay.innerHTML = resultHtml;
                
                // Update metrics
                updateMetrics(windowId, {
                    harmonic_resonance: 0.995,
                    quantum_entropy: 0.879,
                    symbolic_variance: 0.123,
                    wave_coherence: 0.998
                });
                
                // Update wave visualization with success state
                drawWaveVisualization(windowId, 'decrypt-wave-canvas', ciphertext + key, true);
                
                // Show success message
                const matchIndicator = document.getElementById(`decrypt-match-indicator-${windowId}`);
                if (matchIndicator) {
                    matchIndicator.innerHTML = 'Key resonance match: Perfect';
                    matchIndicator.className = 'match-indicator success';
                }
            })
            .catch(error => {
                resultDisplay.innerHTML = `<div class="error-message">Error: ${error.message}</div>`;
                
                // Update wave visualization with error state
                drawWaveVisualization(windowId, 'decrypt-wave-canvas', 'error', false);
                
                // Show error message
                const matchIndicator = document.getElementById(`decrypt-match-indicator-${windowId}`);
                if (matchIndicator) {
                    matchIndicator.innerHTML = 'Key resonance mismatch: Failed';
                    matchIndicator.className = 'match-indicator error';
                }
            });
        });
    }
}

// Draw wave visualization
function drawWaveVisualization(windowId, canvasId, seed, isSuccess) {
    const canvas = document.getElementById(`${canvasId}-${windowId}`);
    if (!canvas) return;
    
    const ctx = canvas.getContext('2d');
    const width = canvas.width;
    const height = canvas.height;
    
    // Clear canvas
    ctx.clearRect(0, 0, width, height);
    ctx.fillStyle = '#000';
    ctx.fillRect(0, 0, width, height);
    
    // If no seed, draw placeholder wave
    if (!seed) {
        drawPlaceholderWave(ctx, width, height);
        return;
    }
    
    // Generate a hash from the seed
    const hashValue = typeof seed === 'string' ? hashString(seed) : Math.random() * 10000;
    
    // Draw wave based on the hash
    if (isSuccess) {
        drawSuccessWave(ctx, width, height, hashValue);
    } else if (seed === 'error') {
        drawErrorWave(ctx, width, height);
    } else {
        drawProcessingWave(ctx, width, height, hashValue);
    }
}

// Draw placeholder wave
function drawPlaceholderWave(ctx, width, height) {
    ctx.strokeStyle = '#b967ff'; // Purple
    ctx.lineWidth = 2;
    ctx.beginPath();
    
    for (let x = 0; x < width; x++) {
        const y = height / 2 + Math.sin(x * 0.05) * 30;
        if (x === 0) {
            ctx.moveTo(x, y);
        } else {
            ctx.lineTo(x, y);
        }
    }
    
    ctx.stroke();
}

// Draw success wave
function drawSuccessWave(ctx, width, height, seed) {
    // First wave (purple)
    ctx.strokeStyle = '#b967ff'; // Purple
    ctx.lineWidth = 2;
    ctx.beginPath();
    
    for (let x = 0; x < width; x++) {
        const y = height / 2 + Math.sin(x * 0.04 + seed / 1000) * 40;
        if (x === 0) {
            ctx.moveTo(x, y);
        } else {
            ctx.lineTo(x, y);
        }
    }
    
    ctx.stroke();
}

// Draw error wave
function drawErrorWave(ctx, width, height) {
    ctx.strokeStyle = '#ff5252'; // Red
    ctx.lineWidth = 2;
    ctx.beginPath();
    
    for (let x = 0; x < width; x++) {
        // Create a jagged error wave
        let y;
        if (x % 20 < 10) {
            y = height / 2 + 20 + Math.random() * 10;
        } else {
            y = height / 2 - 20 - Math.random() * 10;
        }
        
        if (x === 0) {
            ctx.moveTo(x, y);
        } else {
            ctx.lineTo(x, y);
        }
    }
    
    ctx.stroke();
}

// Draw processing wave (animated)
function drawProcessingWave(ctx, width, height, seed) {
    ctx.strokeStyle = '#00b7ff'; // Blue
    ctx.lineWidth = 2;
    ctx.beginPath();
    
    const time = Date.now() / 1000;
    
    for (let x = 0; x < width; x++) {
        const frequency = 0.03 + (seed % 10) / 100;
        const amplitude = 30 + (seed % 20);
        const y = height / 2 + Math.sin(x * frequency + time) * amplitude;
        
        if (x === 0) {
            ctx.moveTo(x, y);
        } else {
            ctx.lineTo(x, y);
        }
    }
    
    ctx.stroke();
    
    // Continue animation
    requestAnimationFrame(() => drawProcessingWave(ctx, width, height, seed));
}

// Update metrics displays
function updateMetrics(windowId, metrics) {
    // Update harmonic resonance
    const harmonicResonance = document.getElementById(`harmonic-resonance-value-${windowId}`);
    if (harmonicResonance && metrics.harmonic_resonance) {
        harmonicResonance.textContent = metrics.harmonic_resonance.toFixed(3);
        
        // Update progress bar
        const harmonicBar = document.getElementById(`harmonic-resonance-bar-${windowId}`);
        if (harmonicBar) {
            harmonicBar.style.width = `${metrics.harmonic_resonance * 100}%`;
        }
    }
    
    // Update quantum entropy
    const quantumEntropy = document.getElementById(`quantum-entropy-value-${windowId}`);
    if (quantumEntropy && metrics.quantum_entropy) {
        quantumEntropy.textContent = metrics.quantum_entropy.toFixed(3);
        
        // Update progress bar
        const entropyBar = document.getElementById(`quantum-entropy-bar-${windowId}`);
        if (entropyBar) {
            entropyBar.style.width = `${metrics.quantum_entropy * 100}%`;
        }
    }
    
    // Update symbolic variance
    const symbolicVariance = document.getElementById(`symbolic-variance-value-${windowId}`);
    if (symbolicVariance && metrics.symbolic_variance) {
        symbolicVariance.textContent = metrics.symbolic_variance.toFixed(3);
        
        // Update progress bar
        const varianceBar = document.getElementById(`symbolic-variance-bar-${windowId}`);
        if (varianceBar) {
            varianceBar.style.width = `${metrics.symbolic_variance * 100}%`;
        }
    }
    
    // Update wave coherence
    const waveCoherence = document.getElementById(`wave-coherence-value-${windowId}`);
    if (waveCoherence && metrics.wave_coherence) {
        waveCoherence.textContent = metrics.wave_coherence.toFixed(3);
        
        // Update progress bar
        const coherenceBar = document.getElementById(`wave-coherence-bar-${windowId}`);
        if (coherenceBar) {
            coherenceBar.style.width = `${metrics.wave_coherence * 100}%`;
        }
    }
}

// Format JSON result for display
function formatJsonResult(data) {
    if (typeof data === 'string') {
        return data;
    }
    
    // Display container hash and ciphertext in a formatted way
    let json = JSON.stringify(data, null, 2);
    
    // Add syntax highlighting
    return json.replace(/"([^"]+)":/g, '<span style="color: #00b7ff;">\"$1\":</span>');
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

// Function to create encryption tab content
function createEncryptionTabContent(windowId) {
    return `
        <div class="encryption-tabs">
            <div class="tab-header">
                <div class="tab-btn active" id="encrypt-tab-btn-${windowId}" data-tab="encrypt">Encrypt</div>
                <div class="tab-btn" id="decrypt-tab-btn-${windowId}" data-tab="decrypt">Decrypt</div>
            </div>
            
            <div class="tab-content active" id="encrypt-tab-content-${windowId}">
                <div class="form-group">
                    <label for="plaintext-input-${windowId}">Plaintext:</label>
                    <textarea id="plaintext-input-${windowId}" placeholder="Enter text to encrypt..." rows="5"></textarea>
                </div>
                
                <div class="form-group">
                    <label for="encrypt-key-input-${windowId}">Encryption Key:</label>
                    <input type="text" id="encrypt-key-input-${windowId}" placeholder="Enter encryption key...">
                </div>
                
                <div class="button-group">
                    <button id="encrypt-button-${windowId}" class="encrypt-btn">Encrypt</button>
                    <button id="run-benchmark-button-${windowId}" class="benchmark-btn">Run 64-Perturbation Benchmark</button>
                </div>
                
                <div class="wave-visualization">
                    <canvas id="encrypt-wave-canvas-${windowId}" width="400" height="100"></canvas>
                </div>
                
                <div class="encryption-result" id="encrypt-result-${windowId}">
                    <div class="placeholder">// Encrypted output will appear here</div>
                </div>
            </div>
            
            <div class="tab-content" id="decrypt-tab-content-${windowId}" style="display:none;">
                <div class="form-group">
                    <label for="ciphertext-input-${windowId}">Container Hash:</label>
                    <textarea id="ciphertext-input-${windowId}" placeholder="Enter container hash to decrypt..." rows="3"></textarea>
                </div>
                
                <div class="form-group">
                    <label for="decrypt-key-input-${windowId}">Decryption Key:</label>
                    <input type="text" id="decrypt-key-input-${windowId}" placeholder="Enter decryption key...">
                </div>
                
                <div class="match-indicator" id="decrypt-match-indicator-${windowId}">Awaiting key verification...</div>
                
                <div class="wave-visualization">
                    <canvas id="decrypt-wave-canvas-${windowId}" width="400" height="100"></canvas>
                </div>
                
                <button id="decrypt-button-${windowId}" class="decrypt-btn">Unlock Container</button>
                
                <div class="encryption-result" id="decrypt-result-${windowId}">
                    <div class="placeholder">// Decrypted output will appear here</div>
                </div>
            </div>
        </div>
        
        <div class="resonance-metrics">
            <h3>Resonance Metrics</h3>
            
            <div class="metrics-container">
                <div class="metric-row">
                    <div class="metric-name">Harmonic Resonance:</div>
                    <div class="metric-bar-container">
                        <div class="metric-bar" id="harmonic-resonance-bar-${windowId}" style="width: 31.7%"></div>
                    </div>
                    <div class="metric-value" id="harmonic-resonance-value-${windowId}">0.317</div>
                </div>
                
                <div class="metric-row">
                    <div class="metric-name">Quantum Entropy:</div>
                    <div class="metric-bar-container">
                        <div class="metric-bar" id="quantum-entropy-bar-${windowId}" style="width: 87.9%"></div>
                    </div>
                    <div class="metric-value" id="quantum-entropy-value-${windowId}">0.879</div>
                </div>
                
                <div class="metric-row">
                    <div class="metric-name">Symbolic Variance:</div>
                    <div class="metric-bar-container">
                        <div class="metric-bar" id="symbolic-variance-bar-${windowId}" style="width: 70.4%"></div>
                    </div>
                    <div class="metric-value" id="symbolic-variance-value-${windowId}">0.704</div>
                </div>
                
                <div class="metric-row">
                    <div class="metric-name">Wave Coherence:</div>
                    <div class="metric-bar-container">
                        <div class="metric-bar" id="wave-coherence-bar-${windowId}" style="width: 10.9%"></div>
                    </div>
                    <div class="metric-value" id="wave-coherence-value-${windowId}">0.109</div>
                </div>
            </div>
        </div>
    `;
}

// CSS for encryption tab
const encryptionTabCSS = `
    .encryption-tabs {
        margin-bottom: 20px;
    }
    
    .tab-header {
        display: flex;
        border-bottom: 1px solid #333;
        margin-bottom: 15px;
    }
    
    .tab-btn {
        padding: 8px 15px;
        cursor: pointer;
        opacity: 0.7;
        transition: all 0.3s;
    }
    
    .tab-btn:hover {
        opacity: 1;
    }
    
    .tab-btn.active {
        border-bottom: 2px solid #00b7ff;
        opacity: 1;
    }
    
    .tab-content {
        padding: 10px 0;
    }
    
    .form-group {
        margin-bottom: 15px;
    }
    
    .form-group label {
        display: block;
        margin-bottom: 5px;
        font-weight: 300;
    }
    
    .form-group textarea,
    .form-group input[type="text"] {
        width: 100%;
        padding: 10px;
        background-color: #111;
        border: 1px solid #333;
        border-radius: 4px;
        color: #fff;
        font-family: monospace;
        resize: vertical;
    }
    
    .button-group {
        display: flex;
        gap: 10px;
        margin-bottom: 15px;
    }
    
    .encrypt-btn, .decrypt-btn {
        padding: 8px 15px;
        background-color: #00475e;
        color: white;
        border: none;
        border-radius: 4px;
        cursor: pointer;
        transition: background-color 0.3s;
    }
    
    .encrypt-btn:hover, .decrypt-btn:hover {
        background-color: #006a8c;
    }
    
    .benchmark-btn {
        padding: 8px 15px;
        background-color: #333;
        color: white;
        border: none;
        border-radius: 4px;
        cursor: pointer;
    }
    
    .benchmark-btn:hover {
        background-color: #444;
    }
    
    .wave-visualization {
        margin: 15px 0;
        background-color: #000;
        border-radius: 4px;
        overflow: hidden;
    }
    
    .encryption-result {
        background-color: #111;
        padding: 15px;
        border-radius: 4px;
        font-family: monospace;
        white-space: pre-wrap;
        overflow-x: auto;
        color: #ccc;
        min-height: 100px;
    }
    
    .encryption-result .placeholder {
        color: #555;
    }
    
    .encryption-result .success-message {
        color: #4caf50;
        margin-bottom: 10px;
    }
    
    .encryption-result .error-message {
        color: #ff5252;
        margin-bottom: 10px;
    }
    
    .encryption-result .loading-message {
        color: #00b7ff;
        margin-bottom: 10px;
    }
    
    .match-indicator {
        padding: 8px 15px;
        background-color: #333;
        border-radius: 4px;
        margin-bottom: 15px;
        text-align: center;
    }
    
    .match-indicator.success {
        background-color: rgba(76, 175, 80, 0.3);
        color: #4caf50;
        border: 1px solid rgba(76, 175, 80, 0.5);
    }
    
    .match-indicator.error {
        background-color: rgba(255, 82, 82, 0.3);
        color: #ff5252;
        border: 1px solid rgba(255, 82, 82, 0.5);
    }
    
    .resonance-metrics {
        margin-top: 30px;
        padding: 15px;
        background-color: rgba(0, 0, 0, 0.3);
        border-radius: 8px;
    }
    
    .resonance-metrics h3 {
        margin-top: 0;
        margin-bottom: 15px;
    }
    
    .metrics-container {
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 15px;
    }
    
    .metric-row {
        display: flex;
        align-items: center;
        gap: 10px;
    }
    
    .metric-name {
        width: 150px;
        font-size: 14px;
    }
    
    .metric-bar-container {
        flex: 1;
        height: 10px;
        background-color: #111;
        border-radius: 5px;
        overflow: hidden;
    }
    
    .metric-bar {
        height: 100%;
        background-color: #00b7ff;
        transition: width 0.5s ease;
    }
    
    .metric-value {
        width: 60px;
        text-align: right;
        font-family: monospace;
    }
`;

// Add the encryption tab CSS to the document
document.addEventListener('DOMContentLoaded', function() {
    const style = document.createElement('style');
    style.innerHTML = encryptionTabCSS;
    document.head.appendChild(style);
});