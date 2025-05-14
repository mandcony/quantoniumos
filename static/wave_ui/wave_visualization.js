/**
 * Quantonium OS - Wave Visualization Module
 * 
 * This module implements the waveform matching visualization for container unlocking
 */

// Make sure we're in the global scope
(function(window) {
    "use strict";
    
    // Function to update the waveform visualization based on hash and key
    function updateWaveformVisualization(containerId, hashValue, keyValue) {
        const container = document.getElementById(containerId);
        if (!container) return;
        
        // Clear any previous visualization
        container.innerHTML = '';
        
        try {
            // Create SVG elements for wave visualization
            const hashSvg = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
            const keySvg = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
            
            hashSvg.setAttribute('viewBox', '0 0 400 100');
            keySvg.setAttribute('viewBox', '0 0 400 100');
            
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
            container.appendChild(hashWaveDiv);
            container.appendChild(hashLabel);
            container.appendChild(keyWaveDiv);
            container.appendChild(keyLabel);
            
            // Generate unique paths based on hash and key
            const hashPath = document.createElementNS('http://www.w3.org/2000/svg', 'path');
            hashPath.setAttribute('class', 'wave-path');
            hashPath.setAttribute('d', generateWaveformPath(hashValue, 100, 400));
            
            const keyPath = document.createElementNS('http://www.w3.org/2000/svg', 'path');
            keyPath.setAttribute('class', 'wave-path');
            keyPath.setAttribute('d', generateWaveformPath(keyValue, 100, 400));
            
            // Add paths to SVGs
            hashSvg.appendChild(hashPath);
            keySvg.appendChild(keyPath);
            
            // Determine if there's a match (simplified algorithm)
            const hashCode1 = hashCode(hashValue);
            const hashCode2 = hashCode(keyValue);
            const match = Math.abs(hashCode1 - hashCode2) < 1000000; // Arbitrary threshold
            
            // Update visualization based on match status
            if (match) {
                hashWaveDiv.classList.add('matched');
                keyWaveDiv.classList.add('matched');
            } else {
                hashWaveDiv.classList.add('mismatched');
                keyWaveDiv.classList.add('mismatched');
            }
            
            return match;
        } catch (error) {
            console.error('Error creating wave visualization:', error);
            return false;
        }
    }
    
    // Function to generate a waveform path from a string input
    function generateWaveformPath(input, height, width) {
        // Convert string to a sequence of values
        let hash = 0;
        for (let i = 0; i < input.length; i++) {
            hash = ((hash << 5) - hash) + input.charCodeAt(i);
            hash |= 0; // Convert to 32bit integer
        }
        
        // Seed a simple random number generator
        const seed = Math.abs(hash);
        const rng = new function() {
            let s = seed;
            return function() {
                s = Math.sin(s) * 10000;
                return s - Math.floor(s);
            };
        };
        
        // Generate points for the path
        const points = [];
        const segments = 20; // Number of segments in the path
        const segmentWidth = width / segments;
        
        for (let i = 0; i <= segments; i++) {
            const x = i * segmentWidth;
            const yOffset = height * 0.4 * rng(); // Random y value
            points.push([x, height / 2 + yOffset]);
        }
        
        // Generate SVG path
        let path = `M ${points[0][0]},${points[0][1]}`;
        
        for (let i = 1; i < points.length; i++) {
            const [x, y] = points[i];
            const [prevX, prevY] = points[i - 1];
            const cpx1 = prevX + (x - prevX) / 3;
            const cpx2 = prevX + (x - prevX) * 2 / 3;
            path += ` C ${cpx1},${prevY} ${cpx2},${y} ${x},${y}`;
        }
        
        return path;
    }
    
    // Helper function to generate a wave path
    function generateWavePath(width, height, complexity = 3, seed = null) {
        // If seed is provided, use it to create a deterministic wave
        const rng = seed ? new function() {
            let s = seed;
            return function() {
                s = Math.sin(s) * 10000;
                return s - Math.floor(s);
            };
        } : Math.random;
        
        // Generate control points
        const points = [];
        const segments = 10 * complexity;
        const segmentWidth = width / segments;
        
        for (let i = 0; i <= segments; i++) {
            const x = i * segmentWidth;
            // More complex waves have more variation
            const yOffset = height * 0.4 * rng();
            points.push([x, height / 2 + yOffset]);
        }
        
        // Generate SVG path
        let path = `M ${points[0][0]},${points[0][1]}`;
        
        for (let i = 1; i < points.length; i++) {
            const [x, y] = points[i];
            const [prevX, prevY] = points[i - 1];
            const cpx1 = prevX + (x - prevX) / 3;
            const cpx2 = prevX + (x - prevX) * 2 / 3;
            path += ` C ${cpx1},${prevY} ${cpx2},${y} ${x},${y}`;
        }
        
        return path;
    }
    
    // Simple hash function for strings
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