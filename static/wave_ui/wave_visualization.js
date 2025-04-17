/**
 * Quantonium OS - Wave Visualization Module
 * 
 * This module implements the waveform matching visualization for container unlocking
 */

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
        
        // Determine if they match by comparing string equality or a hash-based matching algorithm
        // For demo purposes, we'll use a simple algorithm: if the first 3 chars of key match with any 
        // substring in the hash, we consider it a potential match
        const isMatch = hash.includes(key.substring(0, 3)) || key.includes(hash.substring(0, 3));
        
        // Update visualization based on match status
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