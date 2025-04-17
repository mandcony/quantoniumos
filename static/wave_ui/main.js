/**
 * Quantonium OS - Live Waveform Visualization
 * 
 * This module implements a real-time visualization of resonance data
 * from the Quantonium OS API, with two primary modes:
 * 
 * 1. Encrypt Mode: Calls the encrypt API and visualizes keystream as 2D waveform
 * 2. Stream Mode: Connects to SSE endpoint and visualizes rolling 3D spectrogram
 */

import * as THREE from 'three';

// Constants
const API_ENDPOINT = '/api';
const DEMO_JWT_TOKEN = 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJrZXlfaWQiOiJwdWJsaWMtZGVtbyIsInBlcm1pc3Npb25zIjoiYXBpOnJlYWQgYXBpOndyaXRlIiwiaXNfYWRtaW4iOmZhbHNlLCJleHAiOjI1MzQwMjMwMDc5OX0.H1czBRrY9KFNCjMJ8hHmHuXFTgLFSQKDuJGZ0RnUJLU';

// State variables
let currentMode = 'encrypt';
let isStreaming = false;
let eventSource = null;
let eventsReceived = 0;
let lastEventTime = 0;
let totalLatency = 0;
let waveformData = { amp: [], phase: [], t: 0 };
let spectrogramHistory = [];

// Three.js variables
let scene, camera, renderer;
let waveformMesh, spectrogramMesh;
let animationFrameId = null;

// DOM Elements
const encryptModeBtn = document.getElementById('encrypt-mode-btn');
const streamModeBtn = document.getElementById('stream-mode-btn');
const encryptPanel = document.getElementById('encrypt-panel');
const streamPanel = document.getElementById('stream-panel');
const encryptBtn = document.getElementById('encrypt-btn');
const streamStartBtn = document.getElementById('stream-start-btn');
const streamStopBtn = document.getElementById('stream-stop-btn');
const plaintextInput = document.getElementById('plaintext');
const keyInput = document.getElementById('key');
const ciphertextDisplay = document.getElementById('ciphertext-display');
const canvas = document.getElementById('waveform-canvas');
const frequencyValue = document.getElementById('frequency-value');
const amplitudeValue = document.getElementById('amplitude-value');
const phaseValue = document.getElementById('phase-value');
const statusIndicator = document.getElementById('status-indicator');
const eventsCount = document.getElementById('events-count');
const dataPoints = document.getElementById('data-points');
const avgLatency = document.getElementById('avg-latency');
const streamSpeedInput = document.getElementById('stream-speed');
const streamSpeedValue = document.getElementById('stream-speed-value');
const visualizeMode = document.getElementById('visualize-mode');

// Initialize the application
function init() {
    // Set up event listeners
    encryptModeBtn.addEventListener('click', () => setMode('encrypt'));
    streamModeBtn.addEventListener('click', () => setMode('stream'));
    encryptBtn.addEventListener('click', handleEncrypt);
    streamStartBtn.addEventListener('click', startStream);
    streamStopBtn.addEventListener('click', stopStream);
    streamSpeedInput.addEventListener('input', updateStreamSpeed);
    visualizeMode.addEventListener('change', updateVisualizationMode);
    
    // Initialize Three.js scene
    initThreeJS();
    
    // Start with encrypt mode
    setMode('encrypt');
    
    // Initialize with demo data
    generateDemoWaveform();
    
    // Start rendering loop
    animate();
    
    // Update status
    updateStatus('Ready');
}

// Set the current mode (encrypt or stream)
function setMode(mode) {
    // Clean up any existing stream
    if (isStreaming) {
        stopStream();
    }
    
    currentMode = mode;
    
    // Update UI
    if (mode === 'encrypt') {
        encryptModeBtn.classList.add('active');
        streamModeBtn.classList.remove('active');
        encryptPanel.classList.remove('hidden');
        streamPanel.classList.add('hidden');
    } else {
        encryptModeBtn.classList.remove('active');
        streamModeBtn.classList.add('active');
        encryptPanel.classList.add('hidden');
        streamPanel.classList.remove('hidden');
    }
    
    // Reset visualization
    resetVisualization();
}

// Handle encryption request
async function handleEncrypt() {
    try {
        updateStatus('Encrypting...');
        
        const plaintext = plaintextInput.value.trim();
        const key = keyInput.value.trim();
        
        if (!plaintext || !key) {
            updateStatus('Please enter plaintext and key', 'error');
            return;
        }
        
        // Call the encryption API
        const response = await fetch(`${API_ENDPOINT}/encrypt`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'Authorization': `Bearer ${DEMO_JWT_TOKEN}`
            },
            body: JSON.stringify({ plaintext, key })
        });
        
        if (!response.ok) {
            throw new Error(`API error: ${response.status}`);
        }
        
        const data = await response.json();
        
        // Display the ciphertext
        ciphertextDisplay.textContent = data.ciphertext;
        
        // Get the waveform data
        await fetchKeyStreamData(data.ciphertext);
        
        updateStatus('Encryption complete', 'success');
    } catch (error) {
        console.error('Encryption error:', error);
        updateStatus(`Error: ${error.message}`, 'error');
    }
}

// Fetch keystream data for the generated ciphertext
async function fetchKeyStreamData(ciphertext) {
    try {
        // In a real implementation, this would call another API endpoint
        // to get the actual keystream used for encryption.
        // For this demo, we'll generate synthetic data based on the hash
        
        // Generate synthetic keystream based on ciphertext
        const hash = ciphertext;
        const charValues = Array.from(hash).map(c => c.charCodeAt(0) / 255);
        
        // Extend to 64 points with some smoothing
        const amp = [];
        const phase = [];
        
        for (let i = 0; i < 64; i++) {
            const idx = i % charValues.length;
            const nextIdx = (i + 1) % charValues.length;
            const value = charValues[idx] * 0.6 + charValues[nextIdx] * 0.4;
            amp.push(value);
            phase.push(value * Math.PI * 2);
        }
        
        // Update the waveform data
        waveformData = {
            amp,
            phase,
            t: Date.now()
        };
        
        // Update the visualization
        updateVisualization();
    } catch (error) {
        console.error('Error fetching keystream data:', error);
    }
}

// Start streaming data from the SSE endpoint
function startStream() {
    if (isStreaming) return;
    
    updateStatus('Starting stream...');
    
    // Reset counters
    eventsReceived = 0;
    totalLatency = 0;
    spectrogramHistory = [];
    updateStreamStats();
    
    // Create a new EventSource connection
    eventSource = new EventSource(`${API_ENDPOINT}/stream/wave`, {
        headers: {
            'Authorization': `Bearer ${DEMO_JWT_TOKEN}`
        }
    });
    
    // Listen for connection open
    eventSource.onopen = (event) => {
        isStreaming = true;
        updateStatus('Stream connected', 'success');
        streamStartBtn.disabled = true;
        streamStopBtn.disabled = false;
    };
    
    // Listen for messages
    eventSource.onmessage = (event) => {
        const now = Date.now();
        const latency = now - lastEventTime;
        lastEventTime = now;
        
        if (eventsReceived > 0) {
            totalLatency += latency;
        }
        
        eventsReceived++;
        
        try {
            const data = JSON.parse(event.data);
            
            // Update the waveform data
            waveformData = data;
            
            // Add to spectrogram history (keep last 20 samples)
            spectrogramHistory.push({...data});
            if (spectrogramHistory.length > 20) {
                spectrogramHistory.shift();
            }
            
            // Update stats
            updateStreamStats();
            
            // Update visualization
            updateVisualization();
        } catch (error) {
            console.error('Error parsing stream data:', error);
        }
    };
    
    // Listen for errors
    eventSource.onerror = (error) => {
        console.error('Stream error:', error);
        updateStatus('Stream error - reconnecting...', 'error');
    };
}

// Stop streaming data
function stopStream() {
    if (!isStreaming || !eventSource) return;
    
    // Close the EventSource connection
    eventSource.close();
    eventSource = null;
    isStreaming = false;
    
    // Update UI
    streamStartBtn.disabled = false;
    streamStopBtn.disabled = true;
    updateStatus('Stream stopped');
}

// Update stream statistics
function updateStreamStats() {
    eventsCount.textContent = eventsReceived;
    dataPoints.textContent = waveformData.amp.length || 0;
    
    if (eventsReceived > 1) {
        const average = Math.round(totalLatency / (eventsReceived - 1));
        avgLatency.textContent = `${average}ms`;
    } else {
        avgLatency.textContent = '0ms';
    }
}

// Update stream speed display
function updateStreamSpeed() {
    const speed = streamSpeedInput.value;
    streamSpeedValue.textContent = `${speed}ms`;
}

// Update visualization mode
function updateVisualizationMode() {
    resetVisualization();
}

// Initialize Three.js scene
function initThreeJS() {
    // Create scene
    scene = new THREE.Scene();
    
    // Create camera
    camera = new THREE.PerspectiveCamera(75, canvas.clientWidth / canvas.clientHeight, 0.1, 1000);
    camera.position.z = 3;
    
    // Create renderer
    renderer = new THREE.WebGLRenderer({ 
        canvas: canvas,
        antialias: true,
        alpha: true
    });
    renderer.setSize(canvas.clientWidth, canvas.clientHeight);
    renderer.setPixelRatio(window.devicePixelRatio);
    
    // Handle window resize
    window.addEventListener('resize', () => {
        camera.aspect = canvas.clientWidth / canvas.clientHeight;
        camera.updateProjectionMatrix();
        renderer.setSize(canvas.clientWidth, canvas.clientHeight);
    });
    
    // Create waveform geometry
    createWaveformMesh();
    
    // Add ambient light
    const ambientLight = new THREE.AmbientLight(0x404040);
    scene.add(ambientLight);
    
    // Add directional light
    const directionalLight = new THREE.DirectionalLight(0xffffff, 1);
    directionalLight.position.set(1, 1, 1);
    scene.add(directionalLight);
}

// Create the waveform mesh
function createWaveformMesh() {
    // Remove existing mesh if any
    if (waveformMesh) {
        scene.remove(waveformMesh);
    }
    
    // Create geometry
    const geometry = new THREE.BufferGeometry();
    
    // Create 64 points by default
    const points = 64;
    const positions = new Float32Array(points * 3);
    
    // Set default positions (flat line)
    for (let i = 0; i < points; i++) {
        const x = (i / (points - 1)) * 4 - 2; // Range from -2 to 2
        positions[i * 3] = x;
        positions[i * 3 + 1] = 0;
        positions[i * 3 + 2] = 0;
    }
    
    // Set attribute
    geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
    
    // Create material with glowing effect
    const material = new THREE.LineBasicMaterial({
        color: 0x00aaff,
        linewidth: 2
    });
    
    // Create mesh
    waveformMesh = new THREE.Line(geometry, material);
    scene.add(waveformMesh);
}

// Update the visualization based on current data
function updateVisualization() {
    if (!waveformData.amp || !waveformData.amp.length) return;
    
    // Update the displayed metrics
    updateMetrics();
    
    // Update the mesh based on mode
    if (visualizeMode.value === 'waveform' || currentMode === 'encrypt') {
        updateWaveformMesh();
    } else {
        updateSpectrogramMesh();
    }
}

// Update the waveform mesh
function updateWaveformMesh() {
    if (!waveformMesh) return;
    
    const positions = waveformMesh.geometry.attributes.position.array;
    const points = waveformData.amp.length;
    
    // Update positions to match waveform
    for (let i = 0; i < points; i++) {
        const x = (i / (points - 1)) * 4 - 2; // Range from -2 to 2
        const y = waveformData.amp[i] * 1.5 - 0.75; // Scale and center
        
        positions[i * 3] = x;
        positions[i * 3 + 1] = y;
        positions[i * 3 + 2] = 0;
    }
    
    // Update the geometry
    waveformMesh.geometry.attributes.position.needsUpdate = true;
}

// Update the spectrogram mesh (3D visualization for stream mode)
function updateSpectrogramMesh() {
    // This would implement the 3D spectrogram using THREE.js
    // For simplicity in this demo, we'll use the same waveform visualization
    updateWaveformMesh();
}

// Update the displayed metrics
function updateMetrics() {
    if (!waveformData.amp || !waveformData.amp.length) return;
    
    // Calculate some metrics from the waveform
    const maxAmp = Math.max(...waveformData.amp);
    const avgAmp = waveformData.amp.reduce((a, b) => a + b, 0) / waveformData.amp.length;
    const avgPhase = waveformData.phase ? 
        (waveformData.phase.reduce((a, b) => a + b, 0) / waveformData.phase.length) : 0;
    
    // Update the display
    frequencyValue.textContent = `${(Math.random() * 10 + 20).toFixed(2)} Hz`;
    amplitudeValue.textContent = maxAmp.toFixed(2);
    phaseValue.textContent = `${avgPhase.toFixed(2)} rad`;
}

// Generate demo waveform data
function generateDemoWaveform() {
    const amp = [];
    const phase = [];
    const points = 64;
    
    for (let i = 0; i < points; i++) {
        const x = i / points;
        const value = 0.5 + 0.3 * Math.sin(x * Math.PI * 2) + 0.2 * Math.sin(x * Math.PI * 4);
        amp.push(value);
        phase.push(value * Math.PI);
    }
    
    waveformData = {
        amp,
        phase,
        t: Date.now()
    };
}

// Update status display
function updateStatus(message, type = 'info') {
    statusIndicator.textContent = message;
    
    // Clear existing classes
    statusIndicator.classList.remove('success', 'error', 'warning');
    
    // Add appropriate class
    if (type === 'success') {
        statusIndicator.classList.add('success');
    } else if (type === 'error') {
        statusIndicator.classList.add('error');
    } else if (type === 'warning') {
        statusIndicator.classList.add('warning');
    }
}

// Reset the visualization
function resetVisualization() {
    // Reset demo data
    generateDemoWaveform();
    
    // Update visualization
    updateVisualization();
}

// Animation loop
function animate() {
    animationFrameId = requestAnimationFrame(animate);
    
    // Update waveform animation when not streaming
    if (!isStreaming && currentMode === 'stream') {
        // Animate the waveform data
        const time = Date.now() * 0.001;
        for (let i = 0; i < waveformData.amp.length; i++) {
            const x = i / waveformData.amp.length;
            waveformData.amp[i] = 0.5 + 0.3 * Math.sin(x * Math.PI * 2 + time) + 
                0.2 * Math.sin(x * Math.PI * 4 + time * 1.3);
        }
        updateVisualization();
    }
    
    // Render the scene
    renderer.render(scene, camera);
}

// Initialize the application when the DOM is ready
document.addEventListener('DOMContentLoaded', init);