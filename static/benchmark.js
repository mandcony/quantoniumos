/**
 * QuantoniumOS Benchmark Module
 * 
 * Implements the 64-Perturbation Benchmark visualization and testing
 * for cryptographic avalanche effect validation.
 */

// Initialize when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    console.log("Document ready - initializing benchmark module");
    // Set up benchmark components if we're on the benchmark tab
    setupBenchmarkTab();
});

/**
 * Set up the benchmark tab and its components
 */
function setupBenchmarkTab() {
    // Find benchmark button
    const benchBtn = document.getElementById('run-benchmark-btn');
    if (!benchBtn) {
        console.log('Benchmark button not found, skipping setup');
        return;
    }

    console.log('Setting up benchmark tab components');
    
    // Add event listener to benchmark button
    benchBtn.addEventListener('click', runBenchmark);
    
    // Set initial state
    const benchmarkOutput = document.getElementById('benchmark-output');
    if (benchmarkOutput) {
        benchmarkOutput.innerHTML = '<p>Click the button to run the 64-Perturbation Benchmark</p>';
    }
    
    // Initialize any charts if Chart.js is available
    setupCharts();
}

/**
 * Initialize Chart.js components if available
 */
function setupCharts() {
    if (typeof Chart === 'undefined') {
        console.log('Chart.js not found, skipping chart setup');
        return;
    }
    
    console.log('Setting up benchmark charts');
    
    // Set default chart options for dark theme
    Chart.defaults.color = '#fff';
    Chart.defaults.borderColor = 'rgba(255, 255, 255, 0.1)';
}

/**
 * Run the 64-Perturbation Benchmark
 */
async function runBenchmark() {
    console.log('Starting 64-Perturbation Benchmark');
    
    // Get benchmark elements
    const benchBtn = document.getElementById('run-benchmark-btn');
    const outputElement = document.getElementById('benchmark-output');
    const progressContainer = document.getElementById('benchmark-progress-container');
    const progressBar = document.getElementById('benchmark-progress-bar');
    const progressText = document.getElementById('benchmark-progress-text');
    const dlLink = document.getElementById('csv-link');
    
    // Validate elements exist
    if (!benchBtn || !outputElement) {
        console.error('Required benchmark elements not found');
        return;
    }
    
    // Get input values
    const basePT = document.getElementById('benchmark-base-pt')?.value || 
                  document.getElementById('plaintext')?.value || 
                  '00000000000000000000000000000000';
    
    const baseKey = document.getElementById('benchmark-base-key')?.value || 
                   document.getElementById('encrypt-key')?.value || 
                   '00000000000000000000000000000000';
    
    // Set UI state
    benchBtn.disabled = true;
    benchBtn.innerText = 'Running Benchmark...';
    outputElement.innerHTML = '<div class="loader"></div><p>Running 64-Perturbation Benchmark...</p>';
    
    // Show progress bar if available
    if (progressContainer && progressBar && progressText) {
        progressContainer.style.display = 'block';
        progressBar.style.width = '10%';
        progressText.textContent = 'Starting benchmark...';
    }
    
    try {
        // Call benchmark API
        const response = await fetch('/api/benchmark', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                plaintext: basePT,
                key: baseKey
            })
        });
        
        // Update progress
        updateProgress(50, 'Processing results...');
        
        // Check response
        if (!response.ok) {
            throw new Error(`Server returned ${response.status}: ${response.statusText}`);
        }
        
        // Parse response
        const data = await response.json();
        console.log('Benchmark data:', data);
        
        // Check data status
        if (data.status !== 'ok') {
            throw new Error(data.detail || 'Server error');
        }
        
        // Update UI with results
        outputElement.innerHTML = `
            <h3>64-Perturbation Benchmark Complete</h3>
            <p>✓ ${data.rows_written || 0} tests completed</p>
            <p>✓ Max ΔWC = ${data.delta_max_wc?.toFixed(4) || 0}</p>
            <p>✓ Max ΔHR = ${data.delta_max_hr?.toFixed(4) || 0}</p>
        `;
        
        // Show download link if available
        if (dlLink && data.csv_url) {
            dlLink.href = data.csv_url;
            dlLink.style.display = 'inline-block';
        }
        
        // Show both results containers if they exist
        const resultsContainer = document.getElementById('results-container');
        if (resultsContainer) {
            resultsContainer.style.display = 'block';
        }
        
        const benchResultsContainer = document.getElementById('benchmark-results-container');
        if (benchResultsContainer) {
            benchResultsContainer.style.display = 'block';
        }
        
        // Update metrics displays if they exist
        updateMetricsDisplay(data);
        
        // Process the benchmark data for visualization if results are included
        if (data.results && Array.isArray(data.results) && data.results.length > 0) {
            visualizeBenchmarkResults(data.results);
        }
        
        // Update progress
        updateProgress(100, 'Benchmark completed!');
    } catch (error) {
        console.error('Benchmark error:', error);
        outputElement.innerHTML = `<div class="error">Error: ${error.message}</div>`;
        
        // Hide progress
        if (progressContainer) {
            progressContainer.style.display = 'none';
        }
    } finally {
        // Re-enable button
        benchBtn.disabled = false;
        benchBtn.innerText = 'Run 64-Perturbation Benchmark';
    }
}

/**
 * Update the progress bar and text
 */
function updateProgress(percent, message) {
    const progressBar = document.getElementById('benchmark-progress-bar');
    const progressText = document.getElementById('benchmark-progress-text');
    
    if (progressBar) {
        progressBar.style.width = `${percent}%`;
    }
    
    if (progressText) {
        progressText.textContent = message || `${percent}%`;
    }
    
    console.log(`Progress: ${percent}% - ${message}`);
}

/**
 * Update metrics displays with benchmark data
 */
function updateMetricsDisplay(data) {
    // Update metrics if they exist
    const elements = {
        'tests-completed': data.rows_written || 0,
        'max-wc-delta': data.delta_max_wc?.toFixed(4) || 0,
        'max-hr-delta': data.delta_max_hr?.toFixed(4) || 0,
        'avalanche-score': (((data.delta_max_wc || 0) + (data.delta_max_hr || 0)) / 2 * 100).toFixed(1) + '%'
    };
    
    // Update each element if it exists
    Object.entries(elements).forEach(([id, value]) => {
        const el = document.getElementById(id);
        if (el) {
            el.textContent = value;
        }
    });
}

/**
 * Visualize benchmark results with charts
 */
function visualizeBenchmarkResults(results) {
    // Skip if Chart.js isn't available
    if (typeof Chart === 'undefined') {
        console.log('Chart.js not available, skipping visualization');
        return;
    }
    
    console.log('Creating benchmark visualizations');
    
    // Calculate bit change percentages if applicable
    const bitChangeData = results
        .filter(r => r.bits_changed !== undefined)
        .map(r => ({
            testNumber: r.test_id || r.test_number || 0,
            testType: r.test_type || 'unknown',
            bitPosition: r.bit_position || 0,
            bitChangePct: (r.bits_changed / 256) * 100
        }));
    
    // Exit if no valid data
    if (bitChangeData.length === 0) {
        console.log('No bit change data available for visualization');
        return;
    }
    
    // Create bar chart for bit changes
    createBitChangeChart(bitChangeData);
    
    // Create avalanche grid visualization
    createAvalancheGrid(bitChangeData);
}

/**
 * Create a bar chart showing bit changes
 */
function createBitChangeChart(data) {
    const chartCanvas = document.getElementById('bit-flip-chart');
    if (!chartCanvas) {
        console.log('Bit flip chart canvas not found');
        return;
    }
    
    // Clear any existing chart
    if (chartCanvas.chart) {
        chartCanvas.chart.destroy();
    }
    
    // Prepare data
    const labels = data.map(d => `Test ${d.testNumber}`);
    const values = data.map(d => d.bitChangePct);
    const colors = values.map(pct => {
        const hue = Math.min(120, Math.max(0, (pct - 30) * 3));
        return `hsl(${hue}, 80%, 40%)`;
    });
    
    // Create chart
    chartCanvas.chart = new Chart(chartCanvas.getContext('2d'), {
        type: 'bar',
        data: {
            labels: labels,
            datasets: [{
                label: 'Bit Change %',
                data: values,
                backgroundColor: colors,
                borderWidth: 1
            }]
        },
        options: {
            responsive: true,
            scales: {
                y: {
                    beginAtZero: true,
                    max: 100,
                    title: {
                        display: true,
                        text: 'Percentage of Bits Changed'
                    }
                },
                x: {
                    title: {
                        display: true,
                        text: 'Test Number'
                    }
                }
            },
            plugins: {
                title: {
                    display: true,
                    text: 'Bit Change Distribution (64-Perturbation Benchmark)'
                },
                tooltip: {
                    callbacks: {
                        label: (context) => {
                            const index = context.dataIndex;
                            const item = data[index];
                            return [
                                `Bit Change: ${item.bitChangePct.toFixed(2)}%`,
                                `Test Type: ${item.testType}`,
                                `Bit Position: ${item.bitPosition}`
                            ];
                        }
                    }
                }
            }
        }
    });
}

/**
 * Create avalanche grid visualization
 */
function createAvalancheGrid(data) {
    const gridContainer = document.getElementById('avalanche-grid');
    if (!gridContainer) {
        console.log('Avalanche grid container not found');
        return;
    }
    
    // Clear existing content
    gridContainer.innerHTML = '';
    
    // Create grid cells
    data.forEach((item, index) => {
        const cell = document.createElement('div');
        cell.className = 'avalanche-cell';
        
        // Calculate color based on bit change percentage
        const hue = Math.min(120, Math.max(0, (item.bitChangePct - 30) * 3));
        
        // Style cell
        cell.style.backgroundColor = `hsl(${hue}, 80%, 40%)`;
        cell.style.width = '48px';
        cell.style.height = '48px';
        cell.style.margin = '2px';
        cell.style.borderRadius = '4px';
        cell.style.display = 'inline-block';
        cell.style.position = 'relative';
        cell.title = `Test ${item.testNumber}: ${item.bitChangePct.toFixed(2)}% bits changed`;
        
        // Add text label
        const label = document.createElement('div');
        label.style.position = 'absolute';
        label.style.top = '50%';
        label.style.left = '50%';
        label.style.transform = 'translate(-50%, -50%)';
        label.style.color = 'white';
        label.style.fontSize = '12px';
        label.style.fontWeight = 'bold';
        label.style.textShadow = '0 0 2px black';
        label.textContent = `${Math.round(item.bitChangePct)}%`;
        
        // Add to DOM
        cell.appendChild(label);
        gridContainer.appendChild(cell);
    });
    
    // Add explanation text
    const explanation = document.createElement('div');
    explanation.style.marginTop = '10px';
    explanation.style.fontSize = '12px';
    explanation.style.color = '#ccc';
    explanation.textContent = 'Each cell represents one test in the 64-Perturbation Benchmark. ' +
        'Color indicates bit change percentage (green = ideal ~50%, red = poor).';
    
    gridContainer.appendChild(explanation);
}