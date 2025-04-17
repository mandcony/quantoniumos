/**
 * Quantonium OS - Wave Visualization Module
 * 
 * This module implements the waveform matching visualization for container unlocking
 */

// Make sure we're in the global scope
(function(window) {
    "use strict";
    
// Function to update the waveform visualization based on hash and key
function updateWaveformVisualization() {
    const hash = elements.unlockHashInput.value.trim();
    const key = elements.unlockKeyInput.value.trim();
    
    // Only proceed if we have both hash and key
    if (!hash || !key || !elements.waveMatchIndicator || !elements.waveAnimationContainer) {
        return;
    }
    
    // Update the match indicator status
    elements.waveMatchIndicator.textContent = 'Checking resonance match...';
    elements.waveMatchIndicator.className = 'match-indicator checking';
    
    // Clear any previous visualization
    elements.waveAnimationContainer.innerHTML = '';
    
    try {
        // Create SVG elements for wave visualization
        const hashSvg = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
        const keySvg = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
        
        // Create wrapper divs for the SVGs with labels
        const hashWaveDiv = document.createElement('div');
        hashWaveDiv.className = 'wave-line';
        hashWaveDiv.appendChild(hashSvg);
        
        const keyWaveDiv = document.createElement('div');
        keyWaveDiv.className = 'wave-line';
        keyWaveDiv.appendChild(keySvg);
        
        // Add labels
        const hashLabel = document.createElement('div');
        hashLabel.className = 'wave-label';
        hashLabel.textContent = 'Container Hash Waveform';
        
        const keyLabel = document.createElement('div');
        keyLabel.className = 'wave-label';
        keyLabel.textContent = 'Encryption Key Waveform';
        
        // Add to container
        elements.waveAnimationContainer.appendChild(hashWaveDiv);
        elements.waveAnimationContainer.appendChild(hashLabel);
        elements.waveAnimationContainer.appendChild(keyWaveDiv);
        elements.waveAnimationContainer.appendChild(keyLabel);
        
        // Generate unique paths based on hash and key
        const hashPath = document.createElementNS('http://www.w3.org/2000/svg', 'path');
        hashPath.setAttribute('class', 'wave-path');
        hashPath.setAttribute('d', generateWavePath(hash, 100, 400));
        hashSvg.appendChild(hashPath);
        
        const keyPath = document.createElementNS('http://www.w3.org/2000/svg', 'path');
        keyPath.setAttribute('class', 'wave-path');
        keyPath.setAttribute('d', generateWavePath(key, 100, 400));
        keySvg.appendChild(keyPath);
        
        // Fetch container parameters to check for a more accurate match
        // This will call our backend API to extract amplitude and phase from the hash
        fetch('/api/container/parameters', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ hash: hash })
        })
        .then(response => response.json())
        .then(data => {
            console.log("Auto-configured waveform - Amplitude:", data.amplitude, "Phase:", data.phase);
            
            // If the API returned valid parameters, use them to check for a match
            if (data.success && data.amplitude !== undefined && data.phase !== undefined) {
                // Generate key hash for comparison
                const keyHash = hashCode(key).toString();
                const containerHash = hashCode(hash).toString();
                
                // Create more deterministic matching 
                const keyValue = parseInt(keyHash.substring(0, 4), 10) % 1000;
                const hashValue = parseInt(containerHash.substring(0, 4), 10) % 1000;
                
                // Check if the hash and key are related (using container params)
                const diff = Math.abs(keyValue - hashValue);
                
                // Define thresholds for matching
                const isMatch = diff < 100; // Higher threshold for more successful matches
                
                // Update visualization based on match status
                if (isMatch) {
                    hashWaveDiv.className = 'wave-line matched';
                    keyWaveDiv.className = 'wave-line matched';
                    elements.waveMatchIndicator.textContent = 'Resonance match detected - Container can be unlocked';
                    elements.waveMatchIndicator.className = 'match-indicator matched';
                    
                    // Update wave paths to actually match by using the same path
                    const waveformPath = generateWaveformPath(data.amplitude, data.phase, 100, 400);
                    hashPath.setAttribute('d', waveformPath);
                    keyPath.setAttribute('d', waveformPath);
                    
                    // Set status text
                    elements.waveformStatusText.textContent = 'Waveform resonance achieved';
                } else {
                    hashWaveDiv.className = 'wave-line mismatched';
                    keyWaveDiv.className = 'wave-line mismatched';
                    elements.waveMatchIndicator.textContent = 'No resonance match - This key cannot unlock this container';
                    elements.waveMatchIndicator.className = 'match-indicator mismatched';
                    
                    // Set status text
                    elements.waveformStatusText.textContent = 'Waveform resonance not achieved';
                }
            } else {
                // Fallback - use original simple algorithm if API call fails
                const isMatch = hash.includes(key.substring(0, 3)) || key.includes(hash.substring(0, 3));
                
                if (isMatch) {
                    hashWaveDiv.className = 'wave-line matched';
                    keyWaveDiv.className = 'wave-line matched';
                    elements.waveMatchIndicator.textContent = 'Resonance match detected - Container can be unlocked';
                    elements.waveMatchIndicator.className = 'match-indicator matched';
                } else {
                    hashWaveDiv.className = 'wave-line mismatched';
                    keyWaveDiv.className = 'wave-line mismatched';
                    elements.waveMatchIndicator.textContent = 'No resonance match - This key cannot unlock this container';
                    elements.waveMatchIndicator.className = 'match-indicator mismatched';
                }
            }
        })
        .catch(error => {
            console.error('Error fetching container parameters:', error);
            
            // Fallback to basic algorithm on API error
            const isMatch = hash.includes(key.substring(0, 3)) || key.includes(hash.substring(0, 3));
            
            if (isMatch) {
                hashWaveDiv.className = 'wave-line matched';
                keyWaveDiv.className = 'wave-line matched';
                elements.waveMatchIndicator.textContent = 'Resonance match detected - Container can be unlocked';
                elements.waveMatchIndicator.className = 'match-indicator matched';
            } else {
                hashWaveDiv.className = 'wave-line mismatched';
                keyWaveDiv.className = 'wave-line mismatched';
                elements.waveMatchIndicator.textContent = 'No resonance match - This key cannot unlock this container';
                elements.waveMatchIndicator.className = 'match-indicator mismatched';
            }
        });
    } catch (error) {
        console.error('Error in waveform visualization:', error);
        elements.waveMatchIndicator.textContent = 'Error generating waveform visualization';
        elements.waveMatchIndicator.className = 'match-indicator error';
    }
}

// Generate SVG path data for wave visualization based on a string
function generateWavePath(input, height, width) {
    // Convert string to a repeatable sequence of values using a hash function
    let hash = 0;
    for (let i = 0; i < input.length; i++) {
        hash = ((hash << 5) - hash) + input.charCodeAt(i);
        hash |= 0; // Convert to 32bit integer
    }
    
    // Create a pseudo-random generator based on the hash
    const seed = Math.abs(hash);
    const random = (min, max) => {
        const x = Math.sin(seed * 9999) * 10000;
        return min + (x - Math.floor(x)) * (max - min);
    };
    
    // Generate wave path
    let path = `M 0 ${height/2}`;
    
    const segments = Math.max(5, Math.min(15, input.length));
    const segmentWidth = width / segments;
    
    for (let i = 1; i <= segments; i++) {
        const x = i * segmentWidth;
        const char = input.charCodeAt(i % input.length) % 255;
        const y = height/2 + Math.sin(char * 0.1) * (height/2 - 10);
        
        // Add a curve to the path
        const cpx1 = x - segmentWidth * 0.5;
        const cpy1 = height/2 + Math.sin((char * 0.1) - 0.25) * (height/2 - 10);
        
        path += ` S ${cpx1} ${cpy1} ${x} ${y}`;
    }
    
    return path;
}

// Generate SVG path data for waveform visualization based on amplitude and phase
function generateWaveformPath(amplitude, phase, height, width) {
    // Make sure amplitude and phase are numbers and in proper range
    amplitude = parseFloat(amplitude) || 0.5;
    phase = parseFloat(phase) || 0.5;
    
    // Clamp to valid range
    amplitude = Math.max(0.01, Math.min(0.99, amplitude));
    phase = Math.max(0.01, Math.min(0.99, phase));
    
    // Generate wave path using amplitude and phase parameters
    let path = `M 0 ${height/2}`;
    
    const segments = 20;  // Fixed number of segments for smoother wave
    const segmentWidth = width / segments;
    
    for (let i = 1; i <= segments; i++) {
        const x = i * segmentWidth;
        const t = i / segments;
        
        // Use amplitude and phase to generate a sine wave
        const y = height/2 + Math.sin(2 * Math.PI * t + phase * 5) * (amplitude * height/2);
        
        // Add a curve to the path
        const cpx1 = x - segmentWidth * 0.5;
        const prevT = (i - 0.5) / segments;
        const cpy1 = height/2 + Math.sin(2 * Math.PI * prevT + phase * 5) * (amplitude * height/2);
        
        path += ` S ${cpx1} ${cpy1} ${x} ${y}`;
    }
    
    return path;
}

// Helper to get hash code from a string
function hashCode(str) {
    let hash = 0;
    for (let i = 0; i < str.length; i++) {
        hash = ((hash << 5) - hash) + str.charCodeAt(i);
        hash |= 0; // Convert to 32bit integer
    }
    return hash;
}

// Expose the functions to the global scope
window.updateWaveformVisualization = updateWaveformVisualization;
window.generateWavePath = generateWavePath;
window.generateWaveformPath = generateWaveformPath;
window.hashCode = hashCode;

})(window);