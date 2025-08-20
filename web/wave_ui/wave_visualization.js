// Advanced Wave Visualization Engine for QuantoniumOS
class WaveVisualization {
    constructor(canvasId, options = {}) {
        this.canvas = document.getElementById(canvasId);
        this.ctx = this.canvas.getContext('2d');
        
        // Default options
        this.options = {
            backgroundColor: '#0a0a0a',
            waveColor: '#00ffff',
            gridColor: '#333333',
            glowIntensity: 10,
            particleCount: 50,
            ...options
        };
        
        this.waves = [];
        this.particles = [];
        this.time = 0;
        this.isAnimating = false;
        
        this.initializeCanvas();
        this.createParticles();
    }
    
    initializeCanvas() {
        // Set canvas size
        this.canvas.width = this.canvas.offsetWidth;
        this.canvas.height = this.canvas.offsetHeight;
        
        // Handle window resize
        window.addEventListener('resize', () => {
            this.canvas.width = this.canvas.offsetWidth;
            this.canvas.height = this.canvas.offsetHeight;
        });
    }
    
    createParticles() {
        this.particles = [];
        for (let i = 0; i < this.options.particleCount; i++) {
            this.particles.push({
                x: Math.random() * this.canvas.width,
                y: Math.random() * this.canvas.height,
                vx: (Math.random() - 0.5) * 0.5,
                vy: (Math.random() - 0.5) * 0.5,
                size: Math.random() * 2 + 1,
                opacity: Math.random() * 0.5 + 0.2,
                phase: Math.random() * Math.PI * 2
            });
        }
    }
    
    addWave(frequency, amplitude, phase = 0, color = null) {
        this.waves.push({
            frequency,
            amplitude,
            phase,
            color: color || this.options.waveColor
        });
    }
    
    clearWaves() {
        this.waves = [];
    }
    
    startAnimation() {
        this.isAnimating = true;
        this.animate();
    }
    
    stopAnimation() {
        this.isAnimating = false;
    }
    
    animate() {
        if (!this.isAnimating) return;
        
        this.time += 0.016; // ~60 FPS
        this.render();
        requestAnimationFrame(() => this.animate());
    }
    
    render() {
        this.clearCanvas();
        this.drawGrid();
        this.updateParticles();
        this.drawParticles();
        this.drawWaves();
        this.drawQuantumField();
    }
    
    clearCanvas() {
        this.ctx.fillStyle = this.options.backgroundColor;
        this.ctx.fillRect(0, 0, this.canvas.width, this.canvas.height);
    }
    
    drawGrid() {
        const gridSize = 50;
        this.ctx.strokeStyle = this.options.gridColor;
        this.ctx.lineWidth = 0.5;
        this.ctx.globalAlpha = 0.3;
        
        // Vertical lines
        for (let x = 0; x < this.canvas.width; x += gridSize) {
            this.ctx.beginPath();
            this.ctx.moveTo(x, 0);
            this.ctx.lineTo(x, this.canvas.height);
            this.ctx.stroke();
        }
        
        // Horizontal lines
        for (let y = 0; y < this.canvas.height; y += gridSize) {
            this.ctx.beginPath();
            this.ctx.moveTo(0, y);
            this.ctx.lineTo(this.canvas.width, y);
            this.ctx.stroke();
        }
        
        this.ctx.globalAlpha = 1;
    }
    
    drawWaves() {
        const centerY = this.canvas.height / 2;
        const width = this.canvas.width;
        
        this.waves.forEach((wave, index) => {
            this.ctx.beginPath();
            this.ctx.strokeStyle = wave.color;
            this.ctx.lineWidth = 2;
            
            // Add glow effect
            this.ctx.shadowColor = wave.color;
            this.ctx.shadowBlur = this.options.glowIntensity;
            
            for (let x = 0; x < width; x++) {
                const normalizedX = (x / width) * 4 * Math.PI;
                const y = centerY + Math.sin(normalizedX * wave.frequency + this.time + wave.phase) * 
                         wave.amplitude * (this.canvas.height / 6);
                
                if (x === 0) {
                    this.ctx.moveTo(x, y);
                } else {
                    this.ctx.lineTo(x, y);
                }
            }
            
            this.ctx.stroke();
            this.ctx.shadowBlur = 0;
        });
    }
    
    updateParticles() {
        this.particles.forEach(particle => {
            // Update position
            particle.x += particle.vx;
            particle.y += particle.vy;
            
            // Apply wave influence
            const waveInfluence = this.waves.reduce((sum, wave) => {
                const distance = Math.sqrt(
                    Math.pow(particle.x - this.canvas.width / 2, 2) +
                    Math.pow(particle.y - this.canvas.height / 2, 2)
                );
                return sum + Math.sin(distance * 0.01 * wave.frequency + this.time + wave.phase) * 0.1;
            }, 0);
            
            particle.y += waveInfluence;
            
            // Wrap around screen
            if (particle.x > this.canvas.width) particle.x = 0;
            if (particle.x < 0) particle.x = this.canvas.width;
            if (particle.y > this.canvas.height) particle.y = 0;
            if (particle.y < 0) particle.y = this.canvas.height;
            
            // Update opacity based on wave phase
            particle.opacity = 0.2 + Math.sin(this.time + particle.phase) * 0.3;
        });
    }
    
    drawParticles() {
        this.particles.forEach(particle => {
            this.ctx.globalAlpha = Math.max(0, particle.opacity);
            this.ctx.fillStyle = this.options.waveColor;
            this.ctx.beginPath();
            this.ctx.arc(particle.x, particle.y, particle.size, 0, Math.PI * 2);
            this.ctx.fill();
        });
        this.ctx.globalAlpha = 1;
    }
    
    drawQuantumField() {
        // Draw quantum field interference patterns
        const centerX = this.canvas.width / 2;
        const centerY = this.canvas.height / 2;
        
        this.ctx.globalAlpha = 0.1;
        
        for (let i = 0; i < 3; i++) {
            const radius = 50 + i * 30 + Math.sin(this.time + i) * 20;
            this.ctx.strokeStyle = this.options.waveColor;
            this.ctx.lineWidth = 1;
            this.ctx.beginPath();
            this.ctx.arc(centerX, centerY, radius, 0, Math.PI * 2);
            this.ctx.stroke();
        }
        
        this.ctx.globalAlpha = 1;
    }
    
    // Public methods for external control
    setWaveFrequency(index, frequency) {
        if (this.waves[index]) {
            this.waves[index].frequency = frequency;
        }
    }
    
    setWaveAmplitude(index, amplitude) {
        if (this.waves[index]) {
            this.waves[index].amplitude = amplitude;
        }
    }
    
    setWaveColor(index, color) {
        if (this.waves[index]) {
            this.waves[index].color = color;
        }
    }
    
    getQuantumState() {
        // Calculate quantum properties based on current wave states
        const coherence = this.waves.reduce((sum, wave) => {
            return sum + Math.cos(this.time * wave.frequency + wave.phase) * wave.amplitude;
        }, 0) / this.waves.length;
        
        const entanglement = this.waves.length > 1 ? 
            Math.abs(Math.sin(this.time * this.waves[0].frequency) * 
                    Math.cos(this.time * this.waves[1].frequency)) : 0;
        
        const superposition = this.waves.reduce((sum, wave) => {
            return sum + Math.sin(this.time * wave.frequency + wave.phase);
        }, 0) / this.waves.length;
        
        return {
            coherence: coherence || 0,
            entanglement: entanglement || 0,
            superposition: superposition || 0,
            waveCount: this.waves.length,
            time: this.time
        };
    }
}

// Quantum Wave Analyzer
class QuantumWaveAnalyzer {
    constructor() {
        this.sampleRate = 60; // FPS
        this.bufferSize = 1024;
        this.buffer = new Float32Array(this.bufferSize);
        this.bufferIndex = 0;
    }
    
    addSample(value) {
        this.buffer[this.bufferIndex] = value;
        this.bufferIndex = (this.bufferIndex + 1) % this.bufferSize;
    }
    
    getFrequencySpectrum() {
        // Simple FFT-like frequency analysis
        const spectrum = [];
        const halfBuffer = this.bufferSize / 2;
        
        for (let k = 0; k < halfBuffer; k++) {
            let real = 0, imag = 0;
            
            for (let n = 0; n < this.bufferSize; n++) {
                const angle = -2 * Math.PI * k * n / this.bufferSize;
                real += this.buffer[n] * Math.cos(angle);
                imag += this.buffer[n] * Math.sin(angle);
            }
            
            spectrum[k] = Math.sqrt(real * real + imag * imag);
        }
        
        return spectrum;
    }
    
    getDominantFrequency() {
        const spectrum = this.getFrequencySpectrum();
        let maxIndex = 0;
        let maxValue = spectrum[0];
        
        for (let i = 1; i < spectrum.length; i++) {
            if (spectrum[i] > maxValue) {
                maxValue = spectrum[i];
                maxIndex = i;
            }
        }
        
        return (maxIndex * this.sampleRate) / this.bufferSize;
    }
}

// Export classes
window.WaveVisualization = WaveVisualization;
window.QuantumWaveAnalyzer = QuantumWaveAnalyzer;
