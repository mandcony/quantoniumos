/**
 * QuantoniumOS Wave Visualization Module
 * 
 * This module handles the visualization of waveforms for the QuantoniumOS
 * quantum resonance encryption system. It creates animated wave patterns
 * that represent container waveforms and their resonance patterns.
 */

/**
 * Creates a smooth wave path based on parameters
 * @param {number} width - Width of the wave visualization
 * @param {number} height - Height of the visualization
 * @param {number} segments - Number of wave segments
 * @param {number} seed - Seed value to generate deterministic wave patterns
 * @returns {string} - SVG path data string
 */
function createWavePath(width, height, segments = 4, seed = Math.random()) {
    // Set the midpoint of the wave vertically
    const midY = height / 2;
    
    // Initialize random generator with seed for deterministic output
    const rand = mulberry32(seed);
    
    // Start point
    let path = `M0,${midY} `;
    
    // Generate smooth wave segments
    const segmentWidth = width / segments;
    
    for (let i = 0; i < segments; i++) {
        // Control points for the curve
        const cp1x = i * segmentWidth + segmentWidth / 3;
        const cp1y = midY + (rand() * height / 2 - height / 4);
        
        const cp2x = i * segmentWidth + 2 * segmentWidth / 3;
        const cp2y = midY + (rand() * height / 2 - height / 4);
        
        // End point of this segment
        const x = (i + 1) * segmentWidth;
        const y = midY + (i === segments - 1 ? 0 : (rand() * height / 3 - height / 6));
        
        // Add cubic Bezier curve
        path += `C${cp1x},${cp1y} ${cp2x},${cp2y} ${x},${y} `;
    }
    
    return path;
}

/**
 * Simple seeded random number generator (Mulberry32)
 * @param {number} seed - Seed value
 * @returns {function} - Function that produces deterministic random values
 */
function mulberry32(seed) {
    return function() {
        seed = seed + 0x6D2B79F5;
        var t = seed;
        t = Math.imul(t ^ t >>> 15, t | 1);
        t ^= t + Math.imul(t ^ t >>> 7, t | 61);
        return ((t ^ t >>> 14) >>> 0) / 4294967296;
    };
}

/**
 * Creates a wave visualization with both container waveform and resonance pattern
 * @param {string} containerId - ID of the container element
 * @param {string|number} result - Result string or number to generate waveform from
 * @param {boolean} isMatched - Whether the waveforms match (affects styling)
 */
function createWaveVisualization(containerId, result, isMatched = true) {
    const container = document.getElementById(containerId);
    if (!container) return;
    
    // Clear any existing content
    container.innerHTML = '';
    
    try {
        // Create primary waveform
        const mainSvg = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
        mainSvg.setAttribute('viewBox', '0 0 400 100');
        mainSvg.setAttribute('width', '100%');
        mainSvg.setAttribute('height', '100%');
        
        // Create matching waveform 
        const resSvg = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
        resSvg.setAttribute('viewBox', '0 0 400 100');
        resSvg.setAttribute('width', '100%');
        resSvg.setAttribute('height', '100%');
        
        // Create wrapper divs for the SVGs with labels
        const mainWaveDiv = document.createElement('div');
        mainWaveDiv.className = `wave-line ${isMatched ? 'matched' : 'mismatched'}`;
        mainWaveDiv.appendChild(mainSvg);
        
        const resWaveDiv = document.createElement('div');
        resWaveDiv.className = `wave-line ${isMatched ? 'matched' : 'mismatched'}`;
        resWaveDiv.appendChild(resSvg);
        
        // Add labels
        const mainLabel = document.createElement('div');
        mainLabel.className = 'wave-label';
        mainLabel.textContent = 'Container Waveform';
        
        const resLabel = document.createElement('div');
        resLabel.className = 'wave-label';
        resLabel.textContent = 'Resonance Pattern';
        
        // Generate unique paths based on result
        const mainPath = document.createElementNS('http://www.w3.org/2000/svg', 'path');
        mainPath.setAttribute('class', 'wave-path');
        
        // Create a complementary pattern for the second wave
        // In a real quantum system, this would be based on phase coherence
        const complementaryPath = document.createElementNS('http://www.w3.org/2000/svg', 'path');
        complementaryPath.setAttribute('class', 'wave-path');
        
        // Add animation to paths for dynamic effect
        const mainAnimation = document.createElementNS('http://www.w3.org/2000/svg', 'animate');
        mainAnimation.setAttribute('attributeName', 'opacity');
        mainAnimation.setAttribute('values', isMatched ? '0.7;1;0.7' : '0.5;0.8;0.5');
        mainAnimation.setAttribute('dur', '2s');
        mainAnimation.setAttribute('repeatCount', 'indefinite');
        mainPath.appendChild(mainAnimation);
        
        const compAnimation = document.createElementNS('http://www.w3.org/2000/svg', 'animate');
        compAnimation.setAttribute('attributeName', 'opacity');
        compAnimation.setAttribute('values', isMatched ? '0.7;1;0.7' : '0.5;0.8;0.5');
        compAnimation.setAttribute('dur', '2.2s'); // Slightly different timing creates resonance effect
        compAnimation.setAttribute('repeatCount', 'indefinite');
        complementaryPath.appendChild(compAnimation);
        
        // Handle different types of inputs
        if (typeof result === 'string') {
            // Create deterministic waveform from string
            let seed = 0;
            for (let i = 0; i < result.length; i++) {
                seed = ((seed << 5) - seed) + result.charCodeAt(i);
                seed |= 0; // Convert to 32bit integer
            }
            
            // More complex wave patterns with different frequencies
            mainPath.setAttribute('d', createWavePath(400, 70, 5, seed / 1000000));
            
            // For matched waves, make the resonance pattern similar but with phase shift
            // For mismatched, make it notably different
            if (isMatched) {
                complementaryPath.setAttribute('d', createWavePath(400, 70, 5, (seed + 1234) / 1000000));
            } else {
                complementaryPath.setAttribute('d', createWavePath(400, 70, 3, (seed * 2.5) / 1000000));
            }
        } else {
            // Use the provided result directly
            mainPath.setAttribute('d', createWavePath(400, 70, 5, result));
            
            if (isMatched) {
                complementaryPath.setAttribute('d', createWavePath(400, 70, 5, result + 0.1));
            } else {
                complementaryPath.setAttribute('d', createWavePath(400, 70, 3, result * 2.5));
            }
        }
        
        // Add paths to SVGs
        mainSvg.appendChild(mainPath);
        resSvg.appendChild(complementaryPath);
        
        // Create div structure for better layout
        const waveContainer = document.createElement('div');
        waveContainer.className = 'visualization-waves';
        
        // First wave group with label
        const firstWaveGroup = document.createElement('div');
        firstWaveGroup.style.marginBottom = '15px';
        firstWaveGroup.appendChild(mainWaveDiv);
        firstWaveGroup.appendChild(mainLabel);
        
        // Second wave group with label
        const secondWaveGroup = document.createElement('div');
        secondWaveGroup.appendChild(resWaveDiv);
        secondWaveGroup.appendChild(resLabel);
        
        // Add to container in the correct order
        waveContainer.appendChild(firstWaveGroup);
        waveContainer.appendChild(secondWaveGroup);
        container.appendChild(waveContainer);
    } catch (error) {
        console.error('Error creating wave visualization:', error);
        // Fallback visualization with simplified but consistent structure
        
        // Create container for visualization
        const waveContainer = document.createElement('div');
        waveContainer.className = 'visualization-waves';
        
        // Create primary wave
        const mainSvg = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
        mainSvg.setAttribute('viewBox', '0 0 400 100');
        mainSvg.setAttribute('width', '100%');
        mainSvg.setAttribute('height', '100%');
        
        // Create path for primary wave
        const mainPath = document.createElementNS('http://www.w3.org/2000/svg', 'path');
        mainPath.setAttribute('d', 'M0,50 Q50,30 100,50 T200,50 T300,50 T400,50');
        mainPath.setAttribute('class', 'wave-path');
        
        // Create wrapper for primary wave
        const mainWaveDiv = document.createElement('div');
        mainWaveDiv.className = `wave-line ${isMatched ? 'matched' : 'mismatched'}`;
        
        // Add main path to SVG
        mainSvg.appendChild(mainPath);
        // Add SVG to wave div
        mainWaveDiv.appendChild(mainSvg);
        
        // Add label for primary wave
        const mainLabel = document.createElement('div');
        mainLabel.className = 'wave-label';
        mainLabel.textContent = 'Container Waveform';
        
        // Group main wave and label
        const mainGroup = document.createElement('div');
        mainGroup.style.marginBottom = '15px';
        mainGroup.appendChild(mainWaveDiv);
        mainGroup.appendChild(mainLabel);
        
        // Add to container
        waveContainer.appendChild(mainGroup);
        
        // Repeat for secondary wave
        const resSvg = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
        resSvg.setAttribute('viewBox', '0 0 400 100');
        resSvg.setAttribute('width', '100%');
        resSvg.setAttribute('height', '100%');
        
        const resPath = document.createElementNS('http://www.w3.org/2000/svg', 'path');
        resPath.setAttribute('d', 'M0,50 Q80,70 160,30 T240,70 T320,30 T400,50');
        resPath.setAttribute('class', 'wave-path');
        
        const resWaveDiv = document.createElement('div');
        resWaveDiv.className = `wave-line ${isMatched ? 'matched' : 'mismatched'}`;
        
        resSvg.appendChild(resPath);
        resWaveDiv.appendChild(resSvg);
        
        const resLabel = document.createElement('div');
        resLabel.className = 'wave-label';
        resLabel.textContent = 'Resonance Pattern';
        
        const resGroup = document.createElement('div');
        resGroup.appendChild(resWaveDiv);
        resGroup.appendChild(resLabel);
        
        waveContainer.appendChild(resGroup);
        
        // Add complete visualization to main container
        container.appendChild(waveContainer);
    }
}

/**
 * Updates the match indicator element with status and message
 * @param {string} elementId - ID of the indicator element
 * @param {string} status - Status (checking, matched, mismatched, error)
 * @param {string} message - Message to display
 */
function updateMatchIndicator(elementId, status, message) {
    const indicator = document.getElementById(elementId);
    if (!indicator) return;
    
    // Clear existing classes
    indicator.classList.remove('checking', 'matched', 'mismatched', 'error');
    
    // Add appropriate class
    indicator.classList.add(status);
    
    // Set text
    indicator.textContent = message;
}