/**
 * QuantoniumOS - Resonance Analyzer
 * Connects to the QuantoniumOS backend for image pattern analysis using
 * the resonance mathematics engine.
 */

document.addEventListener('DOMContentLoaded', function() {
    // Application elements
    const fileInput = document.getElementById('image-upload');
    const analyzeBtn = document.getElementById('analyze-btn');
    const originalImage = document.getElementById('original-image');
    const patternOverlay = document.getElementById('pattern-overlay');
    const resonanceThreshold = document.getElementById('resonance-threshold');
    const thresholdValue = document.getElementById('threshold-value');
    const frequencyBands = document.getElementById('frequency-bands');
    const symmetryDetection = document.getElementById('symmetry-detection');
    const summaryContent = document.getElementById('summary-content');
    const symbolicContent = document.getElementById('symbolic-content');
    const confidenceValue = document.querySelector('.confidence-value');
    const confidencePercent = document.getElementById('confidence-percent');
    const statusIndicator = document.querySelector('.status-indicator');
    const resonanceChart = document.getElementById('resonance-chart');
    const phaseChart = document.getElementById('phase-chart');
    const resonanceTable = document.getElementById('resonance-table').querySelector('tbody');
    const phaseTable = document.getElementById('phase-table').querySelector('tbody');
    
    // Tab management
    const tabButtons = document.querySelectorAll('.tab-btn');
    const tabPanes = document.querySelectorAll('.tab-pane');
    
    // Chart instances
    let resonanceChartInstance = null;
    let phaseChartInstance = null;
    
    // Analysis results storage
    let analysisResults = null;
    
    // Window controls
    const minimizeBtn = document.getElementById('minimize-btn');
    const maximizeBtn = document.getElementById('maximize-btn');
    const closeBtn = document.getElementById('close-btn');
    
    // Initialize the application
    initializeApp();
    
    /**
     * Initialize the application components and event listeners
     */
    function initializeApp() {
        console.log('Initializing Resonance Analyzer application...');
        
        // Set up event listeners
        fileInput.addEventListener('change', handleFileSelect);
        analyzeBtn.addEventListener('click', analyzeImage);
        resonanceThreshold.addEventListener('input', updateThresholdValue);
        
        // Set up tab switching
        tabButtons.forEach(button => {
            button.addEventListener('click', () => {
                // Toggle active class on tabs
                tabButtons.forEach(btn => btn.classList.remove('active'));
                button.classList.add('active');
                
                // Toggle active class on panes
                const tabName = button.getAttribute('data-tab');
                tabPanes.forEach(pane => pane.classList.remove('active'));
                document.getElementById(tabName).classList.add('active');
            });
        });
        
        // Window control handlers
        minimizeBtn.addEventListener('click', minimizeWindow);
        maximizeBtn.addEventListener('click', maximizeWindow);
        closeBtn.addEventListener('click', closeWindow);
        
        // Initialize threshold value display
        updateThresholdValue();
        
        // Check if we have access to the QuantoniumOS API
        checkApiAvailability();
    }
    
    /**
     * Handle file selection for image analysis
     */
    function handleFileSelect(event) {
        const file = event.target.files[0];
        if (file) {
            // Only allow image files
            if (!file.type.match('image.*')) {
                showStatus('Please select an image file', 'error');
                return;
            }
            
            // Display the selected image
            const reader = new FileReader();
            reader.onload = function(e) {
                originalImage.src = e.target.result;
                analyzeBtn.disabled = false;
                clearResults();
                showStatus('Image loaded. Ready for analysis.');
            };
            reader.readAsDataURL(file);
        }
    }
    
    /**
     * Clear previous analysis results
     */
    function clearResults() {
        // Clear overlay
        patternOverlay.innerHTML = '';
        
        // Clear summary
        summaryContent.innerHTML = '<p>Click "Analyze Pattern" to begin processing.</p>';
        
        // Clear symbolic interpretation
        symbolicContent.innerHTML = '<p>Analysis will appear here after processing.</p>';
        
        // Reset confidence indicator
        confidenceValue.style.width = '0%';
        confidencePercent.textContent = '0%';
        
        // Clear tables
        resonanceTable.innerHTML = '<tr><td colspan="3">No data available</td></tr>';
        phaseTable.innerHTML = '<tr><td colspan="3">No data available</td></tr>';
        
        // Destroy previous charts
        if (resonanceChartInstance) {
            resonanceChartInstance.destroy();
        }
        if (phaseChartInstance) {
            phaseChartInstance.destroy();
        }
        
        // Reset analysis results
        analysisResults = null;
    }
    
    /**
     * Analyze the selected image using the QuantoniumOS resonance engine
     */
    function analyzeImage() {
        showStatus('Analyzing image pattern...', 'processing');
        
        // Get analysis parameters
        const threshold = parseFloat(resonanceThreshold.value) / 100;
        const bands = parseInt(frequencyBands.value);
        const symmetryType = symmetryDetection.value;
        
        // Create form data with the image and parameters
        const formData = new FormData();
        formData.append('image', fileInput.files[0]);
        formData.append('threshold', threshold);
        formData.append('frequency_bands', bands);
        formData.append('symmetry_type', symmetryType);
        
        // Call the API for image analysis
        fetch('/api/analyze/resonance', {
            method: 'POST',
            body: formData
        })
        .then(response => {
            if (!response.ok) {
                throw new Error('Network response was not ok');
            }
            return response.json();
        })
        .then(data => {
            // Store analysis results
            analysisResults = data;
            
            // Update UI with results
            displayResults(data);
            showStatus('Analysis complete', 'success');
        })
        .catch(error => {
            console.error('Error analyzing image:', error);
            
            // If API fails, use demo data for testing purposes
            useDemoResults();
            showStatus('Using demo results (API unavailable)', 'warning');
        });
    }
    
    /**
     * Display analysis results in the UI
     */
    function displayResults(data) {
        // Update confidence indicator
        const confidence = data.overall_confidence * 100;
        confidenceValue.style.width = `${confidence}%`;
        confidencePercent.textContent = `${confidence.toFixed(1)}%`;
        
        // Update summary
        updateSummary(data);
        
        // Update symbolic interpretation
        updateSymbolicInterpretation(data);
        
        // Update charts and tables
        updateResonanceChart(data);
        updatePhaseChart(data);
        updateTables(data);
        
        // Update pattern overlay
        updatePatternOverlay(data);
    }
    
    /**
     * Update the summary section with analysis results
     */
    function updateSummary(data) {
        const summary = `
            <p><strong>Geometric Analysis:</strong> ${data.symmetry_score.toFixed(2)} symmetry score</p>
            <p><strong>Detected Features:</strong> ${data.detected_features.join(', ')}</p>
            <p><strong>Resonance Patterns:</strong> ${data.resonance_patterns.length} significant patterns</p>
            <p><strong>Dominant Frequency:</strong> ${data.dominant_frequency.toFixed(3)} Hz</p>
            <p><strong>Phase Coherence:</strong> ${(data.phase_coherence * 100).toFixed(1)}%</p>
        `;
        summaryContent.innerHTML = summary;
    }
    
    /**
     * Update the symbolic interpretation section
     */
    function updateSymbolicInterpretation(data) {
        const interpretation = `
            <h4>Symbolic Analysis</h4>
            <p>${data.interpretation.summary}</p>
            
            <h4>Significant Elements</h4>
            <ul>
                ${data.interpretation.elements.map(element => `<li>${element}</li>`).join('')}
            </ul>
            
            <h4>Resonance Significance</h4>
            <p>${data.interpretation.resonance}</p>
            
            <h4>Geometric Significance</h4>
            <p>${data.interpretation.geometry}</p>
        `;
        symbolicContent.innerHTML = interpretation;
    }
    
    /**
     * Update the resonance chart with frequency data
     */
    function updateResonanceChart(data) {
        // Prepare data for chart
        const frequencies = data.resonance_patterns.map(p => p.frequency);
        const amplitudes = data.resonance_patterns.map(p => p.amplitude);
        const colors = data.resonance_patterns.map(p => {
            // Color based on pattern type
            switch(p.pattern_type) {
                case 'harmonic': return 'rgba(58, 134, 255, 0.8)';
                case 'golden_ratio': return 'rgba(255, 171, 0, 0.8)';
                case 'fibonacci': return 'rgba(0, 200, 83, 0.8)';
                default: return 'rgba(150, 150, 150, 0.8)';
            }
        });
        
        // Create or update chart
        if (resonanceChartInstance) {
            resonanceChartInstance.destroy();
        }
        
        resonanceChartInstance = new Chart(resonanceChart, {
            type: 'bar',
            data: {
                labels: frequencies.map(f => f.toFixed(3)),
                datasets: [{
                    label: 'Resonance Strength',
                    data: amplitudes,
                    backgroundColor: colors,
                    borderColor: colors.map(c => c.replace('0.8', '1')),
                    borderWidth: 1
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    y: {
                        beginAtZero: true,
                        grid: {
                            color: 'rgba(255, 255, 255, 0.1)'
                        }
                    },
                    x: {
                        grid: {
                            color: 'rgba(255, 255, 255, 0.1)'
                        }
                    }
                },
                plugins: {
                    legend: {
                        display: false
                    },
                    tooltip: {
                        callbacks: {
                            afterLabel: function(context) {
                                const index = context.dataIndex;
                                return `Type: ${data.resonance_patterns[index].pattern_type}`;
                            }
                        }
                    }
                }
            }
        });
    }
    
    /**
     * Update the phase relationship chart
     */
    function updatePhaseChart(data) {
        // Prepare data for radar chart
        const phaseRelationships = data.phase_relationships;
        const labels = phaseRelationships.map(p => `${p.components[0]}-${p.components[1]}`);
        const phaseValues = phaseRelationships.map(p => p.phase_difference);
        const significance = phaseRelationships.map(p => p.significance);
        
        // Create or update chart
        if (phaseChartInstance) {
            phaseChartInstance.destroy();
        }
        
        phaseChartInstance = new Chart(phaseChart, {
            type: 'radar',
            data: {
                labels: labels,
                datasets: [{
                    label: 'Phase Difference',
                    data: phaseValues.map(v => v / Math.PI * 180), // Convert to degrees
                    backgroundColor: 'rgba(58, 134, 255, 0.2)',
                    borderColor: 'rgba(58, 134, 255, 0.8)',
                    borderWidth: 2,
                    pointBackgroundColor: 'rgba(58, 134, 255, 1)',
                    pointRadius: 5
                }, {
                    label: 'Significance',
                    data: significance.map(s => s * 180), // Scale to match phase values
                    backgroundColor: 'rgba(0, 200, 83, 0.2)',
                    borderColor: 'rgba(0, 200, 83, 0.8)',
                    borderWidth: 2,
                    pointBackgroundColor: 'rgba(0, 200, 83, 1)',
                    pointRadius: 5
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    r: {
                        angleLines: {
                            color: 'rgba(255, 255, 255, 0.1)'
                        },
                        grid: {
                            color: 'rgba(255, 255, 255, 0.1)'
                        },
                        pointLabels: {
                            color: 'rgba(255, 255, 255, 0.7)'
                        }
                    }
                },
                plugins: {
                    tooltip: {
                        callbacks: {
                            label: function(context) {
                                const datasetLabel = context.dataset.label;
                                const value = context.raw;
                                if (datasetLabel === 'Phase Difference') {
                                    return `Phase: ${value.toFixed(1)}° (${(value / 180 * Math.PI).toFixed(2)} rad)`;
                                } else {
                                    return `Significance: ${(value / 180).toFixed(2)}`;
                                }
                            }
                        }
                    }
                }
            }
        });
    }
    
    /**
     * Update the data tables with analysis results
     */
    function updateTables(data) {
        // Update resonance table
        resonanceTable.innerHTML = '';
        data.resonance_patterns.forEach(pattern => {
            const row = document.createElement('tr');
            row.innerHTML = `
                <td>${pattern.frequency.toFixed(3)}</td>
                <td>${pattern.amplitude.toFixed(2)}</td>
                <td>${pattern.pattern_type}</td>
            `;
            resonanceTable.appendChild(row);
        });
        
        // Update phase table
        phaseTable.innerHTML = '';
        data.phase_relationships.forEach(relation => {
            const row = document.createElement('tr');
            row.innerHTML = `
                <td>${relation.components.join(' - ')}</td>
                <td>${(relation.phase_difference / Math.PI * 180).toFixed(1)}° (${relation.phase_difference.toFixed(2)} rad)</td>
                <td>${relation.significance.toFixed(2)}</td>
            `;
            phaseTable.appendChild(row);
        });
    }
    
    /**
     * Update the pattern overlay visualization
     */
    function updatePatternOverlay(data) {
        patternOverlay.innerHTML = '';
        
        // Get image dimensions
        const imgWidth = originalImage.clientWidth;
        const imgHeight = originalImage.clientHeight;
        
        // Add detected nodes (points of resonance)
        data.detected_points.forEach(point => {
            const node = document.createElement('div');
            node.className = 'pattern-node';
            
            // Scale coordinates to fit the displayed image
            const x = point.x * imgWidth;
            const y = point.y * imgHeight;
            
            node.style.left = `${x}px`;
            node.style.top = `${y}px`;
            
            // Size based on importance
            const size = 6 + (point.importance * 8);
            node.style.width = `${size}px`;
            node.style.height = `${size}px`;
            
            patternOverlay.appendChild(node);
        });
        
        // Add resonance lines
        data.resonance_lines.forEach(line => {
            const lineElem = document.createElement('div');
            lineElem.className = 'resonance-line';
            
            // Calculate line positioning
            const startX = line.start.x * imgWidth;
            const startY = line.start.y * imgHeight;
            const endX = line.end.x * imgWidth;
            const endY = line.end.y * imgHeight;
            
            // Calculate length and angle
            const length = Math.sqrt(Math.pow(endX - startX, 2) + Math.pow(endY - startY, 2));
            const angle = Math.atan2(endY - startY, endX - startX);
            
            // Set line properties
            lineElem.style.left = `${startX}px`;
            lineElem.style.top = `${startY}px`;
            lineElem.style.width = `${length}px`;
            lineElem.style.transform = `rotate(${angle}rad)`;
            
            // Set opacity based on strength
            lineElem.style.opacity = line.strength.toString();
            
            patternOverlay.appendChild(lineElem);
        });
        
        // Add symmetry axes
        data.symmetry_axes.forEach(axis => {
            const axisElem = document.createElement('div');
            axisElem.className = 'symmetry-axis';
            
            // Calculate axis positioning
            const startX = axis.start.x * imgWidth;
            const startY = axis.start.y * imgHeight;
            const endX = axis.end.x * imgWidth;
            const endY = axis.end.y * imgHeight;
            
            // Calculate length and angle
            const length = Math.sqrt(Math.pow(endX - startX, 2) + Math.pow(endY - startY, 2));
            const angle = Math.atan2(endY - startY, endX - startX);
            
            // Set axis properties
            axisElem.style.left = `${startX}px`;
            axisElem.style.top = `${startY}px`;
            axisElem.style.height = `${length}px`;
            axisElem.style.transform = `rotate(${angle + Math.PI/2}rad)`;
            
            patternOverlay.appendChild(axisElem);
        });
    }
    
    /**
     * Update the threshold value display
     */
    function updateThresholdValue() {
        const value = resonanceThreshold.value / 100;
        thresholdValue.textContent = value.toFixed(2);
    }
    
    /**
     * Show status message in the status bar
     */
    function showStatus(message, type = 'info') {
        statusIndicator.textContent = message;
        
        // Reset classes
        statusIndicator.className = 'status-indicator';
        
        // Add type-specific class if needed
        if (type !== 'info') {
            statusIndicator.classList.add(type);
        }
    }
    
    /**
     * Check if the QuantoniumOS API is available
     */
    function checkApiAvailability() {
        fetch('/api/status')
            .then(response => {
                if (response.ok) {
                    return response.json();
                }
                throw new Error('API not available');
            })
            .then(data => {
                if (data.status === 'ok') {
                    showStatus('Connected to QuantoniumOS Resonance Engine', 'success');
                } else {
                    showStatus('Connected to QuantoniumOS but resonance module unavailable', 'warning');
                }
            })
            .catch(error => {
                console.error('API availability check failed:', error);
                showStatus('Unable to connect to QuantoniumOS API', 'error');
            });
    }
    
    /**
     * Window control functions (for OS integration)
     */
    function minimizeWindow() {
        // Send message to parent window (OS)
        window.parent.postMessage({
            action: 'minimizeApp',
            appName: 'ResonanceAnalyzer'
        }, '*');
    }
    
    function maximizeWindow() {
        // Send message to parent window (OS)
        window.parent.postMessage({
            action: 'maximizeApp',
            appName: 'ResonanceAnalyzer'
        }, '*');
    }
    
    function closeWindow() {
        // Send message to parent window (OS)
        window.parent.postMessage({
            action: 'closeApp',
            appName: 'ResonanceAnalyzer'
        }, '*');
    }
    
    /**
     * Use demo results if API is unavailable (for testing only)
     */
    function useDemoResults() {
        // Demo data structured like the API response
        const demoData = {
            symmetry_score: 0.87,
            detected_features: ['Radial Pattern', 'Triangular Elements', 'Central Node'],
            resonance_patterns: [
                { frequency: 0.125, amplitude: 0.92, pattern_type: 'harmonic' },
                { frequency: 0.250, amplitude: 0.78, pattern_type: 'harmonic' },
                { frequency: 0.375, amplitude: 0.45, pattern_type: 'regular' },
                { frequency: 0.618, amplitude: 0.83, pattern_type: 'golden_ratio' },
                { frequency: 0.750, amplitude: 0.51, pattern_type: 'regular' },
                { frequency: 0.875, amplitude: 0.38, pattern_type: 'regular' }
            ],
            dominant_frequency: 0.125,
            phase_coherence: 0.92,
            phase_relationships: [
                { components: ['Center', 'Node 1'], phase_difference: Math.PI / 4, significance: 0.85 },
                { components: ['Node 1', 'Node 2'], phase_difference: Math.PI / 2, significance: 0.78 },
                { components: ['Node 2', 'Node 3'], phase_difference: Math.PI / 2, significance: 0.76 },
                { components: ['Node 3', 'Center'], phase_difference: 3 * Math.PI / 4, significance: 0.88 }
            ],
            interpretation: {
                summary: 'The pattern exhibits strong resonance characteristics with geometric symmetry suggesting deliberate design. The central square element with radiating nodes creates a wave-based computational structure similar to quantum circuit representations.',
                elements: [
                    'Central processing node with nested geometry',
                    'Eight-fold radial symmetry with equidistant nodes',
                    'Triangular elements representing directional energy flow',
                    'Circular boundary with frequency containment properties'
                ],
                resonance: 'The dominant frequency at 0.125 Hz corresponds to an 8-fold pattern, with harmonic reinforcement at 0.25 Hz. The golden ratio frequency (0.618) suggests sophisticated mathematical knowledge in the design.',
                geometry: 'The geometric arrangement shows precise angular relationships with phase differences that sum to approximately 2π, indicating a balanced system designed for stability.'
            },
            detected_points: [
                { x: 0.5, y: 0.5, importance: 1.0 }, // Center
                { x: 0.5, y: 0.25, importance: 0.8 }, // Top
                { x: 0.75, y: 0.5, importance: 0.8 }, // Right
                { x: 0.5, y: 0.75, importance: 0.8 }, // Bottom
                { x: 0.25, y: 0.5, importance: 0.8 }, // Left
                { x: 0.65, y: 0.35, importance: 0.6 }, // Top-Right
                { x: 0.65, y: 0.65, importance: 0.6 }, // Bottom-Right
                { x: 0.35, y: 0.65, importance: 0.6 }, // Bottom-Left
                { x: 0.35, y: 0.35, importance: 0.6 }  // Top-Left
            ],
            resonance_lines: [
                { start: { x: 0.5, y: 0.5 }, end: { x: 0.5, y: 0.25 }, strength: 0.8 },
                { start: { x: 0.5, y: 0.5 }, end: { x: 0.75, y: 0.5 }, strength: 0.8 },
                { start: { x: 0.5, y: 0.5 }, end: { x: 0.5, y: 0.75 }, strength: 0.8 },
                { start: { x: 0.5, y: 0.5 }, end: { x: 0.25, y: 0.5 }, strength: 0.8 },
                { start: { x: 0.5, y: 0.5 }, end: { x: 0.65, y: 0.35 }, strength: 0.6 },
                { start: { x: 0.5, y: 0.5 }, end: { x: 0.65, y: 0.65 }, strength: 0.6 },
                { start: { x: 0.5, y: 0.5 }, end: { x: 0.35, y: 0.65 }, strength: 0.6 },
                { start: { x: 0.5, y: 0.5 }, end: { x: 0.35, y: 0.35 }, strength: 0.6 }
            ],
            symmetry_axes: [
                { start: { x: 0.5, y: 0.25 }, end: { x: 0.5, y: 0.75 } },
                { start: { x: 0.25, y: 0.5 }, end: { x: 0.75, y: 0.5 } }
            ],
            overall_confidence: 0.87
        };
        
        // Use the demo data
        displayResults(demoData);
    }
});