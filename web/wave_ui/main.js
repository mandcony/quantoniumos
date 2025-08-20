// QuantoniumOS Wave UI Main JavaScript
class QuantumWaveInterface {
    constructor() {
        this.isRunning = false;
        this.animationId = null;
        this.startTime = Date.now();
        
        // Wave parameters
        this.frequency = 1.0;
        this.amplitude = 1.0;
        this.waveType = 'sine';
        
        this.initializeElements();
        this.bindEvents();
        this.updateDisplay();
    }
    
    initializeElements() {
        this.canvas = document.getElementById('wave-canvas');
        this.ctx = this.canvas.getContext('2d');
        
        this.frequencySlider = document.getElementById('frequency');
        this.amplitudeSlider = document.getElementById('amplitude');
        this.waveTypeSelect = document.getElementById('wave-type');
        
        this.freqValue = document.getElementById('freq-value');
        this.ampValue = document.getElementById('amp-value');
        
        this.startBtn = document.getElementById('start-wave');
        this.stopBtn = document.getElementById('stop-wave');
        
        this.waveStats = document.getElementById('wave-stats');
        this.quantumProps = document.getElementById('quantum-props');
        
        // Set canvas size
        this.canvas.width = 800;
        this.canvas.height = 400;
    }
    
    bindEvents() {
        this.frequencySlider.addEventListener('input', (e) => {
            this.frequency = parseFloat(e.target.value);
            this.freqValue.textContent = this.frequency.toFixed(1);
            this.updateDisplay();
        });
        
        this.amplitudeSlider.addEventListener('input', (e) => {
            this.amplitude = parseFloat(e.target.value);
            this.ampValue.textContent = this.amplitude.toFixed(1);
            this.updateDisplay();
        });
        
        this.waveTypeSelect.addEventListener('change', (e) => {
            this.waveType = e.target.value;
            this.updateDisplay();
        });
        
        this.startBtn.addEventListener('click', () => this.startWave());
        this.stopBtn.addEventListener('click', () => this.stopWave());
    }
    
    startWave() {
        if (!this.isRunning) {
            this.isRunning = true;
            this.startTime = Date.now();
            this.animate();
            this.startBtn.textContent = 'Running...';
            this.startBtn.disabled = true;
        }
    }
    
    stopWave() {
        this.isRunning = false;
        if (this.animationId) {
            cancelAnimationFrame(this.animationId);
        }
        this.startBtn.textContent = 'Start Wave';
        this.startBtn.disabled = false;
    }
    
    animate() {
        if (!this.isRunning) return;
        
        this.drawWave();
        this.updateStats();
        this.animationId = requestAnimationFrame(() => this.animate());
    }
    
    drawWave() {
        const width = this.canvas.width;
        const height = this.canvas.height;
        const centerY = height / 2;
        
        // Clear canvas
        this.ctx.fillStyle = '#0a0a0a';
        this.ctx.fillRect(0, 0, width, height);
        
        // Draw grid
        this.drawGrid();
        
        // Draw wave
        const time = (Date.now() - this.startTime) / 1000;
        this.ctx.beginPath();
        this.ctx.strokeStyle = '#00ffff';
        this.ctx.lineWidth = 2;
        
        for (let x = 0; x < width; x++) {
            const t = (x / width) * 4 * Math.PI + time * this.frequency;
            let y;
            
            switch (this.waveType) {
                case 'sine':
                    y = Math.sin(t);
                    break;
                case 'cosine':
                    y = Math.cos(t);
                    break;
                case 'quantum':
                    // Quantum superposition wave
                    y = (Math.sin(t) + Math.cos(t * 1.618)) / 2; // Golden ratio
                    break;
                default:
                    y = Math.sin(t);
            }
            
            y = centerY - (y * this.amplitude * (height / 4));
            
            if (x === 0) {
                this.ctx.moveTo(x, y);
            } else {
                this.ctx.lineTo(x, y);
            }
        }
        
        this.ctx.stroke();
        
        // Add glow effect
        this.ctx.shadowColor = '#00ffff';
        this.ctx.shadowBlur = 10;
        this.ctx.stroke();
        this.ctx.shadowBlur = 0;
    }
    
    drawGrid() {
        this.ctx.strokeStyle = '#333333';
        this.ctx.lineWidth = 1;
        
        // Vertical lines
        for (let x = 0; x < this.canvas.width; x += 50) {
            this.ctx.beginPath();
            this.ctx.moveTo(x, 0);
            this.ctx.lineTo(x, this.canvas.height);
            this.ctx.stroke();
        }
        
        // Horizontal lines
        for (let y = 0; y < this.canvas.height; y += 50) {
            this.ctx.beginPath();
            this.ctx.moveTo(0, y);
            this.ctx.lineTo(this.canvas.width, y);
            this.ctx.stroke();
        }
        
        // Center line
        this.ctx.strokeStyle = '#666666';
        this.ctx.lineWidth = 2;
        this.ctx.beginPath();
        this.ctx.moveTo(0, this.canvas.height / 2);
        this.ctx.lineTo(this.canvas.width, this.canvas.height / 2);
        this.ctx.stroke();
    }
    
    updateStats() {
        const time = (Date.now() - this.startTime) / 1000;
        const wavelength = (2 * Math.PI) / this.frequency;
        const period = 1 / this.frequency;
        
        this.waveStats.innerHTML = `
            <div class="stat-line">
                <span class="stat-label">Time:</span>
                <span class="stat-value">${time.toFixed(2)}s</span>
            </div>
            <div class="stat-line">
                <span class="stat-label">Frequency:</span>
                <span class="stat-value">${this.frequency.toFixed(2)} Hz</span>
            </div>
            <div class="stat-line">
                <span class="stat-label">Amplitude:</span>
                <span class="stat-value">${this.amplitude.toFixed(2)}</span>
            </div>
            <div class="stat-line">
                <span class="stat-label">Period:</span>
                <span class="stat-value">${period.toFixed(2)}s</span>
            </div>
            <div class="stat-line">
                <span class="stat-label">Wavelength:</span>
                <span class="stat-value">${wavelength.toFixed(2)}</span>
            </div>
        `;
        
        // Quantum properties
        const coherence = Math.cos(time * this.frequency) * this.amplitude;
        const entanglement = Math.sin(time * this.frequency * 1.618) * 0.5 + 0.5;
        const superposition = (Math.sin(time) + Math.cos(time)) / 2;
        
        this.quantumProps.innerHTML = `
            <div class="stat-line">
                <span class="stat-label">Coherence:</span>
                <span class="stat-value">${coherence.toFixed(4)}</span>
            </div>
            <div class="stat-line">
                <span class="stat-label">Entanglement:</span>
                <span class="stat-value">${entanglement.toFixed(4)}</span>
            </div>
            <div class="stat-line">
                <span class="stat-label">Superposition:</span>
                <span class="stat-value">${superposition.toFixed(4)}</span>
            </div>
            <div class="stat-line">
                <span class="stat-label">Wave Type:</span>
                <span class="stat-value">${this.waveType}</span>
            </div>
            <div class="stat-line">
                <span class="stat-label">Quantum State:</span>
                <span class="stat-value">${this.isRunning ? 'Active' : 'Idle'}</span>
            </div>
        `;
    }
    
    updateDisplay() {
        if (!this.isRunning) {
            this.drawWave();
            this.updateStats();
        }
    }
}

// Initialize when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    window.quantumWave = new QuantumWaveInterface();
    
    // Add background animation
    const body = document.body;
    const bg = document.createElement('div');
    bg.className = 'wave-background';
    body.appendChild(bg);
    
    // Add some floating particles
    for (let i = 0; i < 5; i++) {
        setTimeout(() => {
            const particle = document.createElement('div');
            particle.className = 'quantum-particle';
            particle.style.top = Math.random() * window.innerHeight + 'px';
            particle.style.animationDelay = Math.random() * 4 + 's';
            body.appendChild(particle);
            
            // Remove particle after animation
            setTimeout(() => {
                if (particle.parentNode) {
                    particle.parentNode.removeChild(particle);
                }
            }, 4000);
        }, i * 1000);
    }
});

// Export for global use
window.QuantumWaveInterface = QuantumWaveInterface;
