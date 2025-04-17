/**
 * Quantonium OS - Live Waveform Visualization
 * 
 * This module implements a real-time visualization of resonance data
 * from the Quantonium OS API, with two primary modes:
 * 
 * 1. Encrypt Mode: Calls the encrypt API and visualizes keystream as 2D waveform
 * 2. Stream Mode: Connects to SSE endpoint and visualizes rolling 3D spectrogram
 */

// Configuration
const API_BASE_URL = '/api'; // Base URL for API endpoints
const SAMPLE_COUNT = 64;     // Number of samples in each waveform
const MAX_FPS = 30;          // Maximum frames per second for visualization

// Global state
let currentMode = 'encrypt';  // Current visualization mode (encrypt or stream)
let visualizationMode = '2d'; // Current visualization mode (2d or 3d)
let colorScheme = 'quantum';  // Current color scheme
let eventSource = null;       // EventSource for SSE
let threeJsScene = null;      // Three.js scene for 3D visualization
let waveformMesh = null;      // Three.js mesh for the waveform
let lastFrameTime = 0;        // Last frame time for FPS calculation
let streamActive = false;     // Whether streaming is active
let totalEvents = 0;          // Total number of events received
let latencies = [];           // Array of latencies for calculating average
let waveformData = {          // Current waveform data
    timestamp: 0,
    amplitude: new Array(SAMPLE_COUNT).fill(0.5),
    phase: new Array(SAMPLE_COUNT).fill(0.5),
    metrics: {
        harmonic_resonance: 0,
        quantum_entropy: 0,
        symbolic_variance: 0,
        wave_coherence: 0
    }
};

// References to DOM elements
const elements = {
    encryptModeBtn: document.getElementById('encrypt-mode-btn'),
    streamModeBtn: document.getElementById('stream-mode-btn'),
    encryptPanel: document.getElementById('encrypt-panel'),
    streamPanel: document.getElementById('stream-panel'),
    plaintext: document.getElementById('plaintext'),
    key: document.getElementById('key'),
    encryptBtn: document.getElementById('encrypt-btn'),
    ciphertextDisplay: document.getElementById('ciphertext-display'),
    streamStartBtn: document.getElementById('stream-start-btn'),
    streamStopBtn: document.getElementById('stream-stop-btn'),
    eventsCount: document.getElementById('events-count'),
    avgLatency: document.getElementById('avg-latency'),
    streamSpeed: document.getElementById('stream-speed'),
    viewMode: document.getElementById('view-mode'),
    colorScheme: document.getElementById('color-scheme'),
    canvas: document.getElementById('waveform-canvas'),
    statusMessage: document.getElementById('status-message'),
    // Meters
    harmonicResonanceMeter: document.getElementById('harmonic-resonance-meter'),
    harmonicResonanceValue: document.getElementById('harmonic-resonance-value'),
    quantumEntropyMeter: document.getElementById('quantum-entropy-meter'),
    quantumEntropyValue: document.getElementById('quantum-entropy-value'),
    symbolicVarianceMeter: document.getElementById('symbolic-variance-meter'),
    symbolicVarianceValue: document.getElementById('symbolic-variance-value'),
    waveCohererenceMeter: document.getElementById('wave-coherence-meter'),
    waveCohererenceValue: document.getElementById('wave-coherence-value')
};

// Canvas context
const ctx = elements.canvas.getContext('2d');

// Initialize the application
function init() {
    // Resize the canvas to fit the container
    resizeCanvas();

    // Set up event listeners
    window.addEventListener('resize', resizeCanvas);
    elements.encryptModeBtn.addEventListener('click', () => setMode('encrypt'));
    elements.streamModeBtn.addEventListener('click', () => setMode('stream'));
    elements.encryptBtn.addEventListener('click', handleEncrypt);
    elements.streamStartBtn.addEventListener('click', startStream);
    elements.streamStopBtn.addEventListener('click', stopStream);
    elements.viewMode.addEventListener('change', updateVisualizationMode);
    elements.colorScheme.addEventListener('change', () => {
        colorScheme = elements.colorScheme.value;
    });

    // Initialize ThreeJS for 3D visualization
    initThreeJS();

    // Generate demo waveform
    generateDemoWaveform();

    // Start the animation loop
    animate();

    // Update UI to reflect current mode
    setMode(currentMode);
}

function resizeCanvas() {
    // Get the visualization container dimensions
    const container = elements.canvas.parentElement;
    const width = container.clientWidth;
    const height = container.clientHeight;
    
    // Set canvas dimensions
    elements.canvas.width = width;
    elements.canvas.height = height;
    
    // If using ThreeJS, update the renderer
    if (threeJsScene && threeJsScene.renderer) {
        threeJsScene.renderer.setSize(width, height);
        threeJsScene.camera.aspect = width / height;
        threeJsScene.camera.updateProjectionMatrix();
    }
}

function setMode(mode) {
    currentMode = mode;
    
    // Update button states
    elements.encryptModeBtn.classList.toggle('active', mode === 'encrypt');
    elements.streamModeBtn.classList.toggle('active', mode === 'stream');
    
    // Show/hide panels
    elements.encryptPanel.classList.toggle('active', mode === 'encrypt');
    elements.streamPanel.classList.toggle('active', mode === 'stream');
    
    // If switching to stream mode and stream is not active, stop the stream
    if (mode !== 'stream' && streamActive) {
        stopStream();
    }
    
    updateStatus(`Mode switched to ${mode}`, 'info');
}

async function handleEncrypt() {
    const plaintext = elements.plaintext.value.trim();
    const key = elements.key.value.trim();
    
    if (!plaintext) {
        updateStatus('Please enter plaintext to encrypt', 'error');
        return;
    }
    
    if (!key) {
        updateStatus('Please enter an encryption key', 'error');
        return;
    }
    
    try {
        elements.encryptBtn.disabled = true;
        updateStatus('Encrypting...', 'info');
        
        // Call the encrypt endpoint
        const response = await fetch(`${API_BASE_URL}/encrypt`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                plaintext,
                key
            })
        });
        
        if (!response.ok) {
            throw new Error(`Encryption failed: ${response.statusText}`);
        }
        
        const data = await response.json();
        elements.ciphertextDisplay.textContent = data.ciphertext;
        updateStatus('Encryption successful', 'success');
        
        // Generate keystream visualization
        await fetchKeyStreamData(data.ciphertext);
        
    } catch (error) {
        console.error('Encryption error:', error);
        updateStatus(`Error: ${error.message}`, 'error');
    } finally {
        elements.encryptBtn.disabled = false;
    }
}

async function fetchKeyStreamData(ciphertext) {
    try {
        // In a real implementation, this would fetch the actual keystream
        // from an API endpoint that reveals the waveform used for encryption
        
        // For this demo, we just generate a deterministic waveform
        // based on the ciphertext to simulate the keystream
        const seed = Array.from(ciphertext).reduce((sum, char) => sum + char.charCodeAt(0), 0);
        const amplitude = [];
        const phase = [];
        
        // Generate pseudo-random but deterministic values based on the ciphertext
        for (let i = 0; i < SAMPLE_COUNT; i++) {
            const t = i / SAMPLE_COUNT;
            amplitude.push(0.2 + 0.6 * Math.sin(seed * t + i / 3));
            phase.push(0.2 + 0.6 * Math.cos(seed * t + i / 5));
        }
        
        // Update waveform data
        waveformData = {
            timestamp: Date.now(),
            amplitude,
            phase,
            metrics: {
                harmonic_resonance: Math.abs(Math.sin(seed * 0.01)) * 0.8 + 0.1,
                quantum_entropy: Math.abs(Math.cos(seed * 0.02)) * 0.8 + 0.1,
                symbolic_variance: Math.abs(Math.sin(seed * 0.03)) * 0.8 + 0.1,
                wave_coherence: Math.abs(Math.cos(seed * 0.04)) * 0.8 + 0.1
            }
        };
        
        // Update metrics display
        updateMetrics();
        
    } catch (error) {
        console.error('Error fetching keystream data:', error);
    }
}

function startStream() {
    if (streamActive) return;
    
    try {
        // Update UI
        elements.streamStartBtn.disabled = true;
        elements.streamStopBtn.disabled = false;
        updateStatus('Connecting to stream...', 'info');
        
        // Reset counters
        totalEvents = 0;
        latencies = [];
        elements.eventsCount.textContent = '0';
        elements.avgLatency.textContent = '0ms';
        elements.streamSpeed.textContent = '0 fps';
        
        // Create EventSource for SSE
        eventSource = new EventSource(`${API_BASE_URL}/stream/wave`);
        
        // Set up event handlers
        eventSource.onopen = () => {
            streamActive = true;
            updateStatus('Connected to stream', 'success');
        };
        
        eventSource.onmessage = (event) => {
            // Parse the data
            try {
                const data = JSON.parse(event.data);
                
                // Calculate latency
                const now = Date.now();
                const latency = now - data.timestamp;
                latencies.push(latency);
                if (latencies.length > 20) latencies.shift();
                
                // Update waveform data
                waveformData = data;
                
                // Update counters
                totalEvents++;
                elements.eventsCount.textContent = totalEvents.toString();
                
                // Update metrics
                updateMetrics();
                updateStreamStats();
                
            } catch (error) {
                console.error('Error parsing stream data:', error);
            }
        };
        
        eventSource.onerror = (error) => {
            console.error('Stream error:', error);
            updateStatus('Stream connection error', 'error');
            stopStream();
        };
        
        // Set up a timer to update the stream speed
        setInterval(updateStreamSpeed, 1000);
        
    } catch (error) {
        console.error('Error starting stream:', error);
        updateStatus(`Error: ${error.message}`, 'error');
        stopStream();
    }
}

function stopStream() {
    if (!streamActive && !eventSource) return;
    
    try {
        // Close the EventSource
        if (eventSource) {
            eventSource.close();
            eventSource = null;
        }
        
        // Update UI
        elements.streamStartBtn.disabled = false;
        elements.streamStopBtn.disabled = true;
        streamActive = false;
        updateStatus('Stream stopped', 'info');
        
    } catch (error) {
        console.error('Error stopping stream:', error);
    }
}

function updateStreamStats() {
    // Calculate average latency
    if (latencies.length > 0) {
        const avgLatency = Math.round(
            latencies.reduce((sum, latency) => sum + latency, 0) / latencies.length
        );
        elements.avgLatency.textContent = `${avgLatency}ms`;
    }
}

function updateStreamSpeed() {
    // Count the number of events in the last second
    const now = Date.now();
    const recentLatencies = latencies.filter(timestamp => now - timestamp < 1000);
    const fps = recentLatencies.length || '0';
    elements.streamSpeed.textContent = `${fps} fps`;
}

function updateVisualizationMode() {
    visualizationMode = elements.viewMode.value;
    
    // Reset the visualization if needed
    resetVisualization();
    
    updateStatus(`Visualization mode: ${visualizationMode}`, 'info');
}

function initThreeJS() {
    // Create scene
    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0x121212);
    
    // Create camera
    const camera = new THREE.PerspectiveCamera(
        75, 
        elements.canvas.width / elements.canvas.height, 
        0.1, 
        1000
    );
    camera.position.z = 5;
    
    // Create renderer
    const renderer = new THREE.WebGLRenderer({
        canvas: elements.canvas,
        antialias: true
    });
    renderer.setSize(elements.canvas.width, elements.canvas.height);
    
    // Create controls
    const controls = new THREE.OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.dampingFactor = 0.25;
    controls.rotateSpeed = 0.5;
    
    // Store scene objects
    threeJsScene = { scene, camera, renderer, controls };
    
    // Create waveform mesh
    createWaveformMesh();
}

function createWaveformMesh() {
    if (!threeJsScene) return;
    
    // Create geometry
    const geometry = new THREE.BufferGeometry();
    
    // Create material based on color scheme
    const material = new THREE.MeshPhongMaterial({
        color: getColorForScheme(colorScheme),
        side: THREE.DoubleSide,
        wireframe: false,
        transparent: true,
        opacity: 0.8,
        specular: 0x404040,
        shininess: 30
    });
    
    // Create mesh
    const mesh = new THREE.Mesh(geometry, material);
    threeJsScene.scene.add(mesh);
    
    // Add ambient light
    const ambientLight = new THREE.AmbientLight(0x404040);
    threeJsScene.scene.add(ambientLight);
    
    // Add directional light
    const directionalLight = new THREE.DirectionalLight(0xffffff, 1);
    directionalLight.position.set(0, 1, 1);
    threeJsScene.scene.add(directionalLight);
    
    // Store mesh
    waveformMesh = mesh;
}

function getColorForScheme(scheme) {
    switch (scheme) {
        case 'quantum':
            return 0x6a1b9a;
        case 'resonance':
            return 0x00897b;
        case 'symbolic':
            return 0x1565c0;
        default:
            return 0x00e5ff;
    }
}

function updateVisualization() {
    if (visualizationMode === '2d') {
        // Draw 2D waveform
        draw2DWaveform();
    } else {
        // Update 3D waveform
        if (threeJsScene && waveformMesh) {
            updateWaveformMesh();
            
            // Render the scene
            threeJsScene.controls.update();
            threeJsScene.renderer.render(threeJsScene.scene, threeJsScene.camera);
        }
    }
}

function draw2DWaveform() {
    if (!ctx) return;
    
    // Clear canvas
    ctx.clearRect(0, 0, elements.canvas.width, elements.canvas.height);
    
    // Set up dimensions
    const width = elements.canvas.width;
    const height = elements.canvas.height;
    const padding = 20;
    const graphHeight = height - padding * 2;
    const graphWidth = width - padding * 2;
    
    // Draw background
    ctx.fillStyle = '#1e1e1e';
    ctx.fillRect(0, 0, width, height);
    
    // Draw grid
    ctx.strokeStyle = 'rgba(255, 255, 255, 0.1)';
    ctx.lineWidth = 1;
    
    // Horizontal grid lines
    for (let i = 0; i <= 10; i++) {
        const y = padding + (graphHeight * i / 10);
        ctx.beginPath();
        ctx.moveTo(padding, y);
        ctx.lineTo(width - padding, y);
        ctx.stroke();
    }
    
    // Vertical grid lines
    for (let i = 0; i <= 10; i++) {
        const x = padding + (graphWidth * i / 10);
        ctx.beginPath();
        ctx.moveTo(x, padding);
        ctx.lineTo(x, height - padding);
        ctx.stroke();
    }
    
    // Draw amplitude waveform
    if (waveformData.amplitude && waveformData.amplitude.length > 0) {
        ctx.strokeStyle = getColorForWaveform('amplitude');
        ctx.lineWidth = 2;
        ctx.beginPath();
        
        for (let i = 0; i < waveformData.amplitude.length; i++) {
            const x = padding + (graphWidth * i / (waveformData.amplitude.length - 1));
            const y = padding + graphHeight - (graphHeight * waveformData.amplitude[i]);
            
            if (i === 0) {
                ctx.moveTo(x, y);
            } else {
                ctx.lineTo(x, y);
            }
        }
        
        ctx.stroke();
    }
    
    // Draw phase waveform
    if (waveformData.phase && waveformData.phase.length > 0) {
        ctx.strokeStyle = getColorForWaveform('phase');
        ctx.lineWidth = 2;
        ctx.beginPath();
        
        for (let i = 0; i < waveformData.phase.length; i++) {
            const x = padding + (graphWidth * i / (waveformData.phase.length - 1));
            const y = padding + graphHeight - (graphHeight * waveformData.phase[i]);
            
            if (i === 0) {
                ctx.moveTo(x, y);
            } else {
                ctx.lineTo(x, y);
            }
        }
        
        ctx.stroke();
    }
    
    // Draw legend
    ctx.fillStyle = '#ffffff';
    ctx.font = '12px sans-serif';
    ctx.fillText('Amplitude', padding, padding - 5);
    
    ctx.fillStyle = getColorForWaveform('amplitude');
    ctx.fillRect(padding + 70, padding - 15, 20, 10);
    
    ctx.fillStyle = '#ffffff';
    ctx.fillText('Phase', padding + 100, padding - 5);
    
    ctx.fillStyle = getColorForWaveform('phase');
    ctx.fillRect(padding + 140, padding - 15, 20, 10);
}

function getColorForWaveform(type) {
    switch (colorScheme) {
        case 'quantum':
            return type === 'amplitude' ? '#6a1b9a' : '#2196f3';
        case 'resonance':
            return type === 'amplitude' ? '#00897b' : '#ff9800';
        case 'symbolic':
            return type === 'amplitude' ? '#1565c0' : '#4caf50';
        default:
            return type === 'amplitude' ? '#00e5ff' : '#ff4081';
    }
}

function updateWaveformMesh() {
    if (!waveformMesh || !waveformData.amplitude || !waveformData.phase) return;
    
    // Create vertices and faces
    const vertices = [];
    const indices = [];
    const gridSize = Math.sqrt(SAMPLE_COUNT);
    
    // Create vertices
    for (let i = 0; i < gridSize; i++) {
        for (let j = 0; j < gridSize; j++) {
            const x = (i / (gridSize - 1)) * 4 - 2;
            const z = (j / (gridSize - 1)) * 4 - 2;
            const index = i * gridSize + j;
            const y = (waveformData.amplitude[index] || 0.5) * 2 - 1;
            
            vertices.push(x, y, z);
        }
    }
    
    // Create indices for faces
    for (let i = 0; i < gridSize - 1; i++) {
        for (let j = 0; j < gridSize - 1; j++) {
            const a = i * gridSize + j;
            const b = i * gridSize + j + 1;
            const c = (i + 1) * gridSize + j;
            const d = (i + 1) * gridSize + j + 1;
            
            // First triangle
            indices.push(a, b, c);
            
            // Second triangle
            indices.push(b, d, c);
        }
    }
    
    // Create buffer attributes
    const geometry = waveformMesh.geometry;
    geometry.setAttribute('position', new THREE.Float32BufferAttribute(vertices, 3));
    geometry.setIndex(indices);
    geometry.computeVertexNormals();
    geometry.computeBoundingSphere();
    
    // Update material color
    waveformMesh.material.color.set(getColorForScheme(colorScheme));
}

function updateMetrics() {
    if (!waveformData.metrics) return;
    
    const metrics = waveformData.metrics;
    
    // Update meter values
    updateMeter(elements.harmonicResonanceMeter, elements.harmonicResonanceValue, metrics.harmonic_resonance);
    updateMeter(elements.quantumEntropyMeter, elements.quantumEntropyValue, metrics.quantum_entropy);
    updateMeter(elements.symbolicVarianceMeter, elements.symbolicVarianceValue, metrics.symbolic_variance);
    updateMeter(elements.waveCohererenceMeter, elements.waveCohererenceValue, metrics.wave_coherence);
}

function updateMeter(meterElement, valueElement, value) {
    if (!meterElement || !valueElement) return;
    
    // Update width of meter
    meterElement.style.setProperty('--meter-width', `${value * 100}%`);
    
    // Set content of value
    valueElement.textContent = value.toFixed(3);
}

function generateDemoWaveform() {
    // Generate a demo waveform for initial display
    const amplitude = [];
    const phase = [];
    
    for (let i = 0; i < SAMPLE_COUNT; i++) {
        const t = i / SAMPLE_COUNT;
        amplitude.push(0.5 + 0.3 * Math.sin(t * Math.PI * 4));
        phase.push(0.5 + 0.3 * Math.cos(t * Math.PI * 6));
    }
    
    waveformData = {
        timestamp: Date.now(),
        amplitude,
        phase,
        metrics: {
            harmonic_resonance: 0.76,
            quantum_entropy: 0.42,
            symbolic_variance: 0.85,
            wave_coherence: 0.63
        }
    };
    
    // Update metrics display
    updateMetrics();
}

function updateStatus(message, type = 'info') {
    if (!elements.statusMessage) return;
    
    // Remove existing classes
    elements.statusMessage.classList.remove('error', 'success', 'info');
    
    // Add new class
    elements.statusMessage.classList.add(type);
    
    // Set message
    elements.statusMessage.textContent = message;
}

function resetVisualization() {
    if (visualizationMode === '3d') {
        // Make sure ThreeJS is initialized
        if (!threeJsScene) {
            initThreeJS();
        } else {
            // Update renderer
            threeJsScene.renderer.setSize(elements.canvas.width, elements.canvas.height);
            threeJsScene.camera.aspect = elements.canvas.width / elements.canvas.height;
            threeJsScene.camera.updateProjectionMatrix();
        }
    }
}

function animate() {
    // Calculate FPS
    const now = performance.now();
    const elapsed = now - lastFrameTime;
    
    // Limit FPS
    if (elapsed > 1000 / MAX_FPS) {
        lastFrameTime = now;
        
        // Update visualization
        updateVisualization();
    }
    
    // Request next frame
    requestAnimationFrame(animate);
}

// Add CSS custom property for meter width
document.documentElement.style.setProperty('--meter-width', '0%');
document.head.insertAdjacentHTML('beforeend', `
    <style>
        .meter::before {
            width: var(--meter-width, 0%) !important;
        }
    </style>
`);

// Initialize the application when the DOM is loaded
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
} else {
    init();
}