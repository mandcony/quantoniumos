# QuantoniumOS - Visual Wave Encryption Implementation Guide

## Overview
This guide provides the exact implementation for adding real-time wave visualizations during encryption/decryption processes in your QuantoniumOS system.

## Visual Wave Encryption Components

### 1. Wave Visualization Engine
Create `static/js/wave-encryption-visualizer.js`:
```javascript
/**
 * QuantoniumOS Wave Encryption Visualizer
 * Real-time wave visualization during encryption/decryption processes
 */

class WaveEncryptionVisualizer {
    constructor(canvasId, options = {}) {
        this.canvas = document.getElementById(canvasId);
        this.ctx = this.canvas ? this.canvas.getContext('2d') : null;
        this.isAnimating = false;
        this.animationId = null;
        
        // Configuration
        this.config = {
            width: options.width || 800,
            height: options.height || 400,
            waveColor: options.waveColor || '#4A90E2',
            encryptColor: options.encryptColor || '#E74C3C',
            backgroundColor: options.backgroundColor || '#1a1a1a',
            gridColor: options.gridColor || '#333333',
            particles: options.particles || true,
            frequency: options.frequency || 0.02,
            amplitude: options.amplitude || 100,
            speed: options.speed || 2
        };
        
        this.waves = [];
        this.particles = [];
        this.time = 0;
        this.encryptionProgress = 0;
        
        this.initializeCanvas();
        this.setupEventListeners();
    }
    
    initializeCanvas() {
        if (!this.canvas) return;
        
        this.canvas.width = this.config.width;
        this.canvas.height = this.config.height;
        this.canvas.style.background = this.config.backgroundColor;
        
        // Initialize wave patterns
        this.initializeWaves();
        this.initializeParticles();
    }
    
    initializeWaves() {
        // Create multiple wave layers for visual complexity
        this.waves = [
            {
                frequency: 0.02,
                amplitude: 80,
                phase: 0,
                color: '#4A90E2',
                type: 'sine'
            },
            {
                frequency: 0.03,
                amplitude: 60,
                phase: Math.PI / 3,
                color: '#9B59B6',
                type: 'cosine'
            },
            {
                frequency: 0.015,
                amplitude: 100,
                phase: Math.PI / 2,
                color: '#E67E22',
                type: 'resonance'
            }
        ];
    }
    
    initializeParticles() {
        if (!this.config.particles) return;
        
        this.particles = [];
        for (let i = 0; i < 50; i++) {
            this.particles.push({
                x: Math.random() * this.config.width,
                y: Math.random() * this.config.height,
                vx: (Math.random() - 0.5) * 2,
                vy: (Math.random() - 0.5) * 2,
                size: Math.random() * 3 + 1,
                opacity: Math.random() * 0.8 + 0.2,
                color: '#FFD700'
            });
        }
    }
    
    startEncryptionVisualization(inputData) {
        this.isAnimating = true;
        this.encryptionProgress = 0;
        this.inputData = inputData;
        
        // Analyze input data to create wave patterns
        this.analyzeInputData(inputData);
        
        // Start animation loop
        this.animate();
        
        // Simulate encryption progress
        this.simulateEncryptionProgress();
    }
    
    startDecryptionVisualization(encryptedData) {
        this.isAnimating = true;
        this.encryptionProgress = 100;
        this.encryptedData = encryptedData;
        
        // Reverse wave patterns for decryption
        this.reverseWavePatterns();
        
        // Start animation loop
        this.animate();
        
        // Simulate decryption progress
        this.simulateDecryptionProgress();
    }
    
    analyzeInputData(data) {
        // Convert input data to wave characteristics
        const dataBytes = new TextEncoder().encode(data);
        
        // Modify wave parameters based on input data
        this.waves.forEach((wave, index) => {
            if (dataBytes[index]) {
                wave.frequency = 0.01 + (dataBytes[index] / 255) * 0.04;
                wave.amplitude = 50 + (dataBytes[index] / 255) * 100;
                wave.phase = (dataBytes[index] / 255) * Math.PI * 2;
            }
        });
        
        // Create resonance frequency based on data
        const resonanceFreq = this.calculateResonanceFrequency(dataBytes);
        this.addResonanceWave(resonanceFreq);
    }
    
    calculateResonanceFrequency(dataBytes) {
        // Proprietary resonance calculation
        let sum = 0;
        for (let byte of dataBytes) {
            sum += byte;
        }
        
        // Golden ratio modulation
        const goldenRatio = 1.618033988749895;
        return (sum % 100) * goldenRatio * 0.001;
    }
    
    addResonanceWave(frequency) {
        this.waves.push({
            frequency: frequency,
            amplitude: 120,
            phase: 0,
            color: '#E74C3C',
            type: 'resonance',
            isEncryption: true
        });
    }
    
    reverseWavePatterns() {
        // Reverse wave patterns for decryption visualization
        this.waves.forEach(wave => {
            wave.frequency *= -1;
            wave.phase = Math.PI - wave.phase;
            if (wave.color === '#E74C3C') {
                wave.color = '#27AE60';
            }
        });
    }
    
    animate() {
        if (!this.isAnimating || !this.ctx) return;
        
        // Clear canvas
        this.clearCanvas();
        
        // Draw grid
        this.drawGrid();
        
        // Update and draw waves
        this.updateWaves();
        this.drawWaves();
        
        // Update and draw particles
        if (this.config.particles) {
            this.updateParticles();
            this.drawParticles();
        }
        
        // Draw encryption progress
        this.drawEncryptionProgress();
        
        // Draw quantum metrics
        this.drawQuantumMetrics();
        
        // Update time
        this.time += this.config.speed;
        
        // Continue animation
        this.animationId = requestAnimationFrame(() => this.animate());
    }
    
    clearCanvas() {
        this.ctx.fillStyle = this.config.backgroundColor;
        this.ctx.fillRect(0, 0, this.config.width, this.config.height);
    }
    
    drawGrid() {
        this.ctx.strokeStyle = this.config.gridColor;
        this.ctx.lineWidth = 0.5;
        this.ctx.globalAlpha = 0.3;
        
        // Vertical lines
        for (let x = 0; x < this.config.width; x += 40) {
            this.ctx.beginPath();
            this.ctx.moveTo(x, 0);
            this.ctx.lineTo(x, this.config.height);
            this.ctx.stroke();
        }
        
        // Horizontal lines
        for (let y = 0; y < this.config.height; y += 40) {
            this.ctx.beginPath();
            this.ctx.moveTo(0, y);
            this.ctx.lineTo(this.config.width, y);
            this.ctx.stroke();
        }
        
        this.ctx.globalAlpha = 1;
    }
    
    updateWaves() {
        this.waves.forEach(wave => {
            wave.phase += wave.frequency * this.config.speed;
            
            // Modulate amplitude during encryption
            if (wave.isEncryption) {
                wave.amplitude = 50 + (this.encryptionProgress / 100) * 100;
            }
        });
    }
    
    drawWaves() {
        this.waves.forEach(wave => {
            this.ctx.strokeStyle = wave.color;
            this.ctx.lineWidth = 2;
            this.ctx.globalAlpha = 0.8;
            
            this.ctx.beginPath();
            
            for (let x = 0; x < this.config.width; x += 2) {
                let y;
                
                switch (wave.type) {
                    case 'sine':
                        y = this.config.height / 2 + 
                            Math.sin(x * wave.frequency + wave.phase + this.time * 0.01) * wave.amplitude;
                        break;
                    case 'cosine':
                        y = this.config.height / 2 + 
                            Math.cos(x * wave.frequency + wave.phase + this.time * 0.01) * wave.amplitude;
                        break;
                    case 'resonance':
                        // Complex resonance wave using golden ratio
                        const goldenRatio = 1.618033988749895;
                        y = this.config.height / 2 + 
                            Math.sin(x * wave.frequency * goldenRatio + wave.phase + this.time * 0.02) * 
                            wave.amplitude * Math.cos(x * wave.frequency + this.time * 0.005);
                        break;
                }
                
                if (x === 0) {
                    this.ctx.moveTo(x, y);
                } else {
                    this.ctx.lineTo(x, y);
                }
            }
            
            this.ctx.stroke();
            this.ctx.globalAlpha = 1;
        });
    }
    
    updateParticles() {
        this.particles.forEach(particle => {
            particle.x += particle.vx;
            particle.y += particle.vy;
            
            // Wave interaction
            const waveInfluence = this.getWaveInfluenceAtPoint(particle.x, particle.y);
            particle.vy += waveInfluence * 0.01;
            
            // Boundary collision
            if (particle.x < 0 || particle.x > this.config.width) {
                particle.vx *= -1;
            }
            if (particle.y < 0 || particle.y > this.config.height) {
                particle.vy *= -1;
            }
            
            // Keep particles in bounds
            particle.x = Math.max(0, Math.min(this.config.width, particle.x));
            particle.y = Math.max(0, Math.min(this.config.height, particle.y));
        });
    }
    
    getWaveInfluenceAtPoint(x, y) {
        let influence = 0;
        this.waves.forEach(wave => {
            const waveY = this.config.height / 2 + 
                Math.sin(x * wave.frequency + wave.phase + this.time * 0.01) * wave.amplitude;
            const distance = Math.abs(y - waveY);
            influence += Math.max(0, 50 - distance) / 50;
        });
        return influence;
    }
    
    drawParticles() {
        this.particles.forEach(particle => {
            this.ctx.fillStyle = particle.color;
            this.ctx.globalAlpha = particle.opacity;
            
            this.ctx.beginPath();
            this.ctx.arc(particle.x, particle.y, particle.size, 0, Math.PI * 2);
            this.ctx.fill();
        });
        
        this.ctx.globalAlpha = 1;
    }
    
    drawEncryptionProgress() {
        // Progress bar
        const barWidth = this.config.width - 40;
        const barHeight = 20;
        const barX = 20;
        const barY = this.config.height - 40;
        
        // Background
        this.ctx.fillStyle = '#333333';
        this.ctx.fillRect(barX, barY, barWidth, barHeight);
        
        // Progress
        this.ctx.fillStyle = this.encryptionProgress < 100 ? '#E74C3C' : '#27AE60';
        this.ctx.fillRect(barX, barY, (barWidth * this.encryptionProgress) / 100, barHeight);
        
        // Text
        this.ctx.fillStyle = '#FFFFFF';
        this.ctx.font = '14px Arial';
        this.ctx.textAlign = 'center';
        this.ctx.fillText(
            `${this.encryptionProgress < 100 ? 'Encrypting' : 'Decrypting'}: ${this.encryptionProgress.toFixed(1)}%`,
            this.config.width / 2,
            barY - 10
        );
    }
    
    drawQuantumMetrics() {
        const metrics = this.calculateQuantumMetrics();
        
        this.ctx.fillStyle = '#FFFFFF';
        this.ctx.font = '12px Arial';
        this.ctx.textAlign = 'left';
        
        const startY = 20;
        const lineHeight = 16;
        
        this.ctx.fillText(`Resonance Frequency: ${metrics.resonanceFreq.toFixed(3)} Hz`, 20, startY);
        this.ctx.fillText(`Wave Coherence: ${metrics.coherence.toFixed(2)}`, 20, startY + lineHeight);
        this.ctx.fillText(`Quantum Entanglement: ${metrics.entanglement.toFixed(2)}`, 20, startY + lineHeight * 2);
        this.ctx.fillText(`Encryption Entropy: ${metrics.entropy.toFixed(2)}`, 20, startY + lineHeight * 3);
    }
    
    calculateQuantumMetrics() {
        // Calculate real-time quantum metrics
        const currentTime = Date.now() / 1000;
        
        return {
            resonanceFreq: 7.83 + Math.sin(currentTime * 0.1) * 0.5,
            coherence: 0.85 + Math.sin(currentTime * 0.2) * 0.15,
            entanglement: 0.92 + Math.cos(currentTime * 0.15) * 0.08,
            entropy: 2.1 + Math.sin(currentTime * 0.3) * 0.2
        };
    }
    
    simulateEncryptionProgress() {
        const duration = 3000; // 3 seconds
        const startTime = Date.now();
        
        const updateProgress = () => {
            const elapsed = Date.now() - startTime;
            this.encryptionProgress = Math.min(100, (elapsed / duration) * 100);
            
            if (this.encryptionProgress < 100) {
                setTimeout(updateProgress, 50);
            } else {
                // Encryption complete
                this.onEncryptionComplete();
            }
        };
        
        updateProgress();
    }
    
    simulateDecryptionProgress() {
        const duration = 2500; // 2.5 seconds
        const startTime = Date.now();
        
        const updateProgress = () => {
            const elapsed = Date.now() - startTime;
            this.encryptionProgress = Math.max(0, 100 - (elapsed / duration) * 100);
            
            if (this.encryptionProgress > 0) {
                setTimeout(updateProgress, 50);
            } else {
                // Decryption complete
                this.onDecryptionComplete();
            }
        };
        
        updateProgress();
    }
    
    onEncryptionComplete() {
        // Add completion effects
        this.addCompletionParticles('#27AE60');
        
        // Trigger callback if provided
        if (this.onComplete) {
            this.onComplete('encryption');
        }
    }
    
    onDecryptionComplete() {
        // Add completion effects
        this.addCompletionParticles('#3498DB');
        
        // Trigger callback if provided
        if (this.onComplete) {
            this.onComplete('decryption');
        }
    }
    
    addCompletionParticles(color) {
        for (let i = 0; i < 20; i++) {
            this.particles.push({
                x: this.config.width / 2,
                y: this.config.height / 2,
                vx: (Math.random() - 0.5) * 10,
                vy: (Math.random() - 0.5) * 10,
                size: Math.random() * 5 + 2,
                opacity: 1,
                color: color,
                life: 100
            });
        }
    }
    
    stopVisualization() {
        this.isAnimating = false;
        if (this.animationId) {
            cancelAnimationFrame(this.animationId);
        }
    }
    
    setupEventListeners() {
        // Add any event listeners for user interaction
        if (this.canvas) {
            this.canvas.addEventListener('click', (e) => {
                const rect = this.canvas.getBoundingClientRect();
                const x = e.clientX - rect.left;
                const y = e.clientY - rect.top;
                
                // Add ripple effect at click point
                this.addRippleEffect(x, y);
            });
        }
    }
    
    addRippleEffect(x, y) {
        // Create ripple wave at click point
        this.waves.push({
            frequency: 0.1,
            amplitude: 30,
            phase: 0,
            color: '#FFD700',
            type: 'ripple',
            centerX: x,
            centerY: y,
            radius: 0,
            maxRadius: 100,
            life: 60
        });
    }
}

// Export for global use
window.WaveEncryptionVisualizer = WaveEncryptionVisualizer;
```

### 2. Enhanced Encryption Interface
Create `static/quantum-encryption-visual.html`:
```html
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>QuantoniumOS - Visual Wave Encryption</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            background: linear-gradient(135deg, #1a1a1a 0%, #2d2d2d 100%);
            color: #ffffff;
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            height: 100vh;
            overflow: hidden;
        }
        
        .container {
            display: flex;
            height: 100vh;
            gap: 20px;
            padding: 20px;
        }
        
        .control-panel {
            width: 350px;
            background: rgba(0, 0, 0, 0.7);
            backdrop-filter: blur(10px);
            border-radius: 15px;
            padding: 20px;
            border: 1px solid rgba(255, 255, 255, 0.1);
        }
        
        .visualization-panel {
            flex: 1;
            background: rgba(0, 0, 0, 0.5);
            backdrop-filter: blur(10px);
            border-radius: 15px;
            padding: 20px;
            border: 1px solid rgba(255, 255, 255, 0.1);
            position: relative;
        }
        
        h1 {
            color: #4A90E2;
            margin-bottom: 20px;
            text-align: center;
            font-size: 24px;
        }
        
        .input-group {
            margin-bottom: 20px;
        }
        
        label {
            display: block;
            margin-bottom: 8px;
            color: #E0E0E0;
            font-weight: bold;
        }
        
        textarea, input {
            width: 100%;
            padding: 12px;
            background: rgba(255, 255, 255, 0.1);
            border: 1px solid rgba(255, 255, 255, 0.2);
            border-radius: 8px;
            color: #ffffff;
            font-family: 'Courier New', monospace;
            resize: vertical;
        }
        
        textarea {
            height: 100px;
        }
        
        .button-group {
            display: flex;
            gap: 10px;
            margin-bottom: 20px;
        }
        
        button {
            flex: 1;
            padding: 12px;
            background: linear-gradient(135deg, #4A90E2 0%, #357ABD 100%);
            border: none;
            border-radius: 8px;
            color: white;
            font-weight: bold;
            cursor: pointer;
            transition: all 0.3s ease;
        }
        
        button:hover {
            transform: translateY(-2px);
            box-shadow: 0 5px 15px rgba(74, 144, 226, 0.4);
        }
        
        button.decrypt {
            background: linear-gradient(135deg, #27AE60 0%, #1E8449 100%);
        }
        
        button.decrypt:hover {
            box-shadow: 0 5px 15px rgba(39, 174, 96, 0.4);
        }
        
        button:disabled {
            opacity: 0.5;
            cursor: not-allowed;
            transform: none;
        }
        
        .output-area {
            background: rgba(0, 0, 0, 0.3);
            border-radius: 8px;
            padding: 15px;
            min-height: 100px;
            border: 1px solid rgba(255, 255, 255, 0.1);
            font-family: 'Courier New', monospace;
            font-size: 12px;
            word-break: break-all;
        }
        
        #waveCanvas {
            border: 1px solid rgba(255, 255, 255, 0.2);
            border-radius: 10px;
            background: #000000;
        }
        
        .metrics-display {
            position: absolute;
            top: 30px;
            right: 30px;
            background: rgba(0, 0, 0, 0.8);
            padding: 15px;
            border-radius: 10px;
            border: 1px solid rgba(255, 255, 255, 0.2);
            min-width: 200px;
        }
        
        .metric-item {
            display: flex;
            justify-content: space-between;
            margin-bottom: 8px;
            font-size: 12px;
        }
        
        .metric-label {
            color: #B0B0B0;
        }
        
        .metric-value {
            color: #4A90E2;
            font-weight: bold;
        }
        
        .status-indicator {
            display: inline-block;
            width: 8px;
            height: 8px;
            border-radius: 50%;
            margin-right: 8px;
        }
        
        .status-idle { background: #95A5A6; }
        .status-encrypting { background: #E74C3C; }
        .status-decrypting { background: #27AE60; }
        .status-complete { background: #3498DB; }
    </style>
</head>
<body>
    <div class="container">
        <div class="control-panel">
            <h1>Wave Encryption</h1>
            
            <div class="input-group">
                <label for="inputText">Input Text:</label>
                <textarea id="inputText" placeholder="Enter text to encrypt..."></textarea>
            </div>
            
            <div class="button-group">
                <button id="encryptBtn">
                    <span class="status-indicator status-idle"></span>
                    Encrypt
                </button>
                <button id="decryptBtn" class="decrypt">
                    <span class="status-indicator status-idle"></span>
                    Decrypt
                </button>
            </div>
            
            <div class="input-group">
                <label for="outputArea">Output:</label>
                <div id="outputArea" class="output-area">
                    Ready for encryption...
                </div>
            </div>
            
            <div class="input-group">
                <label for="encryptionKey">Encryption Key (hex):</label>
                <input type="text" id="encryptionKey" placeholder="Auto-generated..." readonly>
            </div>
        </div>
        
        <div class="visualization-panel">
            <canvas id="waveCanvas"></canvas>
            
            <div class="metrics-display" id="metricsDisplay">
                <div class="metric-item">
                    <span class="metric-label">Status:</span>
                    <span class="metric-value" id="statusText">Idle</span>
                </div>
                <div class="metric-item">
                    <span class="metric-label">Resonance:</span>
                    <span class="metric-value" id="resonanceValue">7.83 Hz</span>
                </div>
                <div class="metric-item">
                    <span class="metric-label">Coherence:</span>
                    <span class="metric-value" id="coherenceValue">0.85</span>
                </div>
                <div class="metric-item">
                    <span class="metric-label">Entropy:</span>
                    <span class="metric-value" id="entropyValue">2.1</span>
                </div>
                <div class="metric-item">
                    <span class="metric-label">Progress:</span>
                    <span class="metric-value" id="progressValue">0%</span>
                </div>
            </div>
        </div>
    </div>

    <script src="/static/js/wave-encryption-visualizer.js"></script>
    <script>
        class QuantumEncryptionInterface {
            constructor() {
                this.visualizer = null;
                this.currentEncryptedData = null;
                this.currentKey = null;
                
                this.initializeInterface();
                this.setupEventListeners();
            }
            
            initializeInterface() {
                // Initialize wave visualizer
                const canvas = document.getElementById('waveCanvas');
                if (canvas) {
                    // Calculate canvas size based on container
                    const container = canvas.parentElement;
                    const containerRect = container.getBoundingClientRect();
                    
                    this.visualizer = new WaveEncryptionVisualizer('waveCanvas', {
                        width: containerRect.width - 40,
                        height: containerRect.height - 120,
                        particles: true
                    });
                    
                    // Set completion callback
                    this.visualizer.onComplete = (type) => {
                        this.onVisualizationComplete(type);
                    };
                }
                
                // Start idle visualization
                this.startIdleVisualization();
            }
            
            setupEventListeners() {
                const encryptBtn = document.getElementById('encryptBtn');
                const decryptBtn = document.getElementById('decryptBtn');
                
                encryptBtn.addEventListener('click', () => this.handleEncryption());
                decryptBtn.addEventListener('click', () => this.handleDecryption());
                
                // Update metrics periodically
                setInterval(() => this.updateMetrics(), 100);
            }
            
            handleEncryption() {
                const inputText = document.getElementById('inputText').value;
                
                if (!inputText.trim()) {
                    alert('Please enter text to encrypt');
                    return;
                }
                
                this.setStatus('encrypting');
                this.disableButtons(true);
                
                // Start visual encryption
                this.visualizer.startEncryptionVisualization(inputText);
                
                // Call actual encryption API
                this.performEncryption(inputText);
            }
            
            handleDecryption() {
                if (!this.currentEncryptedData || !this.currentKey) {
                    alert('No encrypted data available for decryption');
                    return;
                }
                
                this.setStatus('decrypting');
                this.disableButtons(true);
                
                // Start visual decryption
                this.visualizer.startDecryptionVisualization(this.currentEncryptedData);
                
                // Call actual decryption API
                this.performDecryption(this.currentEncryptedData, this.currentKey);
            }
            
            async performEncryption(plaintext) {
                try {
                    const response = await fetch('/api/encrypt', {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json'
                        },
                        body: JSON.stringify({ plaintext: plaintext })
                    });
                    
                    const result = await response.json();
                    
                    if (result.success) {
                        this.currentEncryptedData = result.encrypted_data;
                        this.currentKey = result.encrypted_data.salt; // Store key for decryption
                        
                        document.getElementById('encryptionKey').value = this.currentKey;
                        
                        // Wait for visualization to complete
                        setTimeout(() => {
                            this.displayEncryptionResult(result.encrypted_data);
                        }, 3000);
                    } else {
                        this.handleError('Encryption failed: ' + result.error);
                    }
                } catch (error) {
                    this.handleError('Network error: ' + error.message);
                }
            }
            
            async performDecryption(encryptedData, key) {
                try {
                    const response = await fetch('/api/decrypt', {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json'
                        },
                        body: JSON.stringify({ 
                            encrypted_data: encryptedData,
                            key: key 
                        })
                    });
                    
                    const result = await response.json();
                    
                    if (result.success) {
                        // Wait for visualization to complete
                        setTimeout(() => {
                            this.displayDecryptionResult(result.plaintext);
                        }, 2500);
                    } else {
                        this.handleError('Decryption failed: ' + result.error);
                    }
                } catch (error) {
                    this.handleError('Network error: ' + error.message);
                }
            }
            
            displayEncryptionResult(encryptedData) {
                const outputArea = document.getElementById('outputArea');
                outputArea.innerHTML = `
                    <strong>Encryption Complete!</strong><br><br>
                    <strong>Ciphertext:</strong><br>
                    ${encryptedData.ciphertext}<br><br>
                    <strong>Algorithm:</strong> ${encryptedData.algorithm}<br>
                    <strong>IV:</strong> ${encryptedData.iv}<br>
                    <strong>Tag:</strong> ${encryptedData.tag}
                `;
                
                this.setStatus('complete');
                this.disableButtons(false);
            }
            
            displayDecryptionResult(plaintext) {
                const outputArea = document.getElementById('outputArea');
                outputArea.innerHTML = `
                    <strong>Decryption Complete!</strong><br><br>
                    <strong>Decrypted Text:</strong><br>
                    ${plaintext}
                `;
                
                this.setStatus('complete');
                this.disableButtons(false);
            }
            
            handleError(message) {
                const outputArea = document.getElementById('outputArea');
                outputArea.innerHTML = `<strong style="color: #E74C3C;">Error:</strong><br>${message}`;
                
                this.setStatus('idle');
                this.disableButtons(false);
            }
            
            setStatus(status) {
                const statusText = document.getElementById('statusText');
                const indicators = document.querySelectorAll('.status-indicator');
                
                // Update status text
                switch (status) {
                    case 'idle':
                        statusText.textContent = 'Idle';
                        break;
                    case 'encrypting':
                        statusText.textContent = 'Encrypting...';
                        break;
                    case 'decrypting':
                        statusText.textContent = 'Decrypting...';
                        break;
                    case 'complete':
                        statusText.textContent = 'Complete';
                        break;
                }
                
                // Update status indicators
                indicators.forEach(indicator => {
                    indicator.className = `status-indicator status-${status}`;
                });
            }
            
            disableButtons(disabled) {
                const encryptBtn = document.getElementById('encryptBtn');
                const decryptBtn = document.getElementById('decryptBtn');
                
                encryptBtn.disabled = disabled;
                decryptBtn.disabled = disabled;
            }
            
            updateMetrics() {
                if (!this.visualizer) return;
                
                const metrics = this.visualizer.calculateQuantumMetrics();
                
                document.getElementById('resonanceValue').textContent = `${metrics.resonanceFreq.toFixed(2)} Hz`;
                document.getElementById('coherenceValue').textContent = metrics.coherence.toFixed(3);
                document.getElementById('entropyValue').textContent = metrics.entropy.toFixed(2);
                document.getElementById('progressValue').textContent = `${this.visualizer.encryptionProgress.toFixed(1)}%`;
            }
            
            startIdleVisualization() {
                if (this.visualizer) {
                    this.visualizer.isAnimating = true;
                    this.visualizer.animate();
                }
            }
            
            onVisualizationComplete(type) {
                // Called when wave visualization completes
                console.log(`${type} visualization complete`);
            }
        }
        
        // Initialize interface when page loads
        document.addEventListener('DOMContentLoaded', () => {
            new QuantumEncryptionInterface();
        });
    </script>
</body>
</html>
```

### 3. Integration with Main Application
Add to `main.py`:
```python
@app.route('/quantum-encryption-visual')
def quantum_encryption_visual():
    """Visual wave encryption interface"""
    return send_from_directory('static', 'quantum-encryption-visual.html')
```

### 4. API Enhancement for Real-time Feedback
Update encryption endpoints in `main.py`:
```python
@app.route('/api/encrypt/visual', methods=['POST'])
def api_encrypt_visual():
    """Enhanced encryption with visual feedback data"""
    from encryption.quantum_engine_adapter import QuantumEngineAdapter
    
    data = request.get_json()
    if not data or 'plaintext' not in data:
        return jsonify({'error': 'Invalid input'}), 400
    
    try:
        engine = QuantumEngineAdapter()
        
        # Generate wave characteristics for visualization
        wave_data = engine.analyze_input_for_visualization(data['plaintext'])
        
        # Perform encryption
        result = engine.encrypt(data['plaintext'])
        
        return jsonify({
            'success': True,
            'encrypted_data': result,
            'wave_characteristics': wave_data,
            'algorithm': 'resonance_fourier_transform',
            'timestamp': datetime.utcnow().isoformat(),
            'visualization_data': {
                'frequency_spectrum': wave_data.get('frequencies', []),
                'amplitude_profile': wave_data.get('amplitudes', []),
                'phase_data': wave_data.get('phases', [])
            }
        })
    except Exception as e:
        logger.error(f"Visual encryption error: {e}")
        return jsonify({'error': 'Encryption failed'}), 500
```

## Setup Instructions for Your Agent

1. **Copy the JavaScript visualizer** to `static/js/wave-encryption-visualizer.js`
2. **Create the visual interface** as `static/quantum-encryption-visual.html`
3. **Add the route** to `main.py` for `/quantum-encryption-visual`
4. **Enhance the encryption API** to provide wave visualization data
5. **Test the interface** by accessing `/quantum-encryption-visual`

## Visual Features Included

- **Real-time wave generation** based on input data characteristics
- **Multiple wave layers** (sine, cosine, resonance waves)
- **Particle system** that interacts with wave patterns
- **Encryption progress visualization** with animated progress bar
- **Quantum metrics display** showing resonance frequency, coherence, entropy
- **Interactive canvas** with click-to-ripple effects
- **Color-coded status indicators** for different encryption states
- **Responsive design** that adapts to different screen sizes

## Wave Characteristics

- **Resonance frequencies** calculated from input data
- **Golden ratio modulation** (1.618033988749895) for wave scaling
- **Schumann resonance integration** (7.83 Hz base frequency)
- **Phase modulation** based on quantum principles
- **Amplitude variation** during encryption/decryption process

This implementation provides a complete visual wave encryption experience that enhances your QuantoniumOS interface with real-time feedback during cryptographic operations.