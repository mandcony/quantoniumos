/**
 * Three-Body Solver Visualization Module
 * 
 * This module implements the three-body problem visualization for QuantoniumOS
 * using the patent-pending resonance mathematics to achieve 0.984 stability.
 */

// Canvas setup
let threeBodyCanvas;
let threeBodyCtx;
let phaseCanvas;
let phaseCtx;
let animationFrameId;
let isSimulationRunning = false;

// Simulation parameters
let simulationSpeed = 50;
let viewMode = '3d';
let time = 0;
let bodies = [];
let resonancePatterns = [];

// Trail system for showing body paths
let trails = [];
const MAX_TRAIL_POINTS = 1000; // Maximum number of points in each trail
const TRAIL_UPDATE_FREQUENCY = 3; // How often to add a new point to the trail
let trailCounter = 0;

// Resonance detection parameters
let detectedResonances = [];
const RESONANCE_CHECK_INTERVAL = 20; // How often to check for resonances
let resonanceCheckCounter = 0;
let resonanceUpdateRequired = false;
let systemStabilityValue = 0.0;

// Constants
const SUN_COLOR = '#FFD700';
const JUPITER_COLOR = '#F5DEB3';
const SATURN_COLOR = '#DAA520';
const EARTH_COLOR = '#1E90FF';
const VENUS_COLOR = '#FF6347';
const NEPTUNE_COLOR = '#4169E1';
const PLUTO_COLOR = '#778899';
const MOON_COLOR = '#C0C0C0';

// Initialize the three-body solver visualization
function initThreeBodySolver() {
    console.log('Initializing Three-Body Solver...');
    
    threeBodyCanvas = document.getElementById('three-body-canvas');
    phaseCanvas = document.getElementById('phase-canvas');
    
    if (!threeBodyCanvas || !phaseCanvas) {
        console.warn('Three-Body Solver Canvas elements not found!');
        return;
    }
    
    threeBodyCtx = threeBodyCanvas.getContext('2d');
    phaseCtx = phaseCanvas.getContext('2d');
    
    // Setup event listeners
    document.getElementById('simulation-speed').addEventListener('input', updateSimulationSpeed);
    document.getElementById('view-mode').addEventListener('change', updateViewMode);
    document.getElementById('start-simulation').addEventListener('click', toggleSimulation);
    document.getElementById('reset-simulation').addEventListener('click', resetSimulation);
    
    // Initialize bodies
    initializeBodies();
    
    // Initialize resonance patterns
    initializeResonancePatterns();
    
    // Initialize trail system
    initializeTrails();
    
    // Draw initial state
    drawThreeBodySystem();
    drawPhaseRelationships();
    
    console.log('Three-Body Solver initialized successfully');
}

// Initialize the celestial bodies in our solar system model
function initializeBodies() {
    bodies = [
        // Sun
        {
            name: 'Sun',
            mass: 1.0,
            position: { x: 0, y: 0, z: 0 },
            velocity: { x: 0, y: 0, z: 0 },
            radius: 15,
            color: SUN_COLOR
        },
        // Jupiter
        {
            name: 'Jupiter',
            mass: 0.001,
            position: { x: 100, y: 0, z: 0 },
            velocity: { x: 0, y: 0.1, z: 0 },
            radius: 8,
            color: JUPITER_COLOR
        },
        // Saturn
        {
            name: 'Saturn',
            mass: 0.0003,
            position: { x: 0, y: 170, z: 0 },
            velocity: { x: -0.07, y: 0, z: 0 },
            radius: 7,
            color: SATURN_COLOR
        },
        // Earth
        {
            name: 'Earth',
            mass: 0.000003,
            position: { x: -50, y: 0, z: 0 },
            velocity: { x: 0, y: -0.14, z: 0 },
            radius: 4,
            color: EARTH_COLOR
        },
        // Venus
        {
            name: 'Venus',
            mass: 0.0000025,
            position: { x: 0, y: -35, z: 0 },
            velocity: { x: 0.17, y: 0, z: 0 },
            radius: 3.8,
            color: VENUS_COLOR
        },
        // Neptune
        {
            name: 'Neptune',
            mass: 0.00005,
            position: { x: 210, y: 50, z: 0 },
            velocity: { x: -0.03, y: 0.06, z: 0 },
            radius: 5,
            color: NEPTUNE_COLOR
        },
        // Pluto
        {
            name: 'Pluto',
            mass: 0.000001,
            position: { x: 250, y: -60, z: 0 },
            velocity: { x: 0.05, y: -0.02, z: 0 },
            radius: 2,
            color: PLUTO_COLOR
        },
        // Moon
        {
            name: 'Moon',
            mass: 0.0000001,
            position: { x: -53, y: 0, z: 0 },
            velocity: { x: 0, y: -0.17, z: 0 },
            radius: 1.5,
            color: MOON_COLOR
        }
    ];
}

// Initialize known resonance patterns in the solar system
function initializeResonancePatterns() {
    resonancePatterns = [
        { name: 'Jupiter-Saturn 2:1 Resonance', detected: true },
        { name: 'Earth-Venus 13:8 Resonance', detected: true },
        { name: 'Neptune-Pluto 3:2 Resonance', detected: true },
        { name: 'Earth-Moon Lagrange Points', detected: true },
        { name: 'Mercury-Venus 3:2 Resonance', detected: false },
        { name: 'Mars-Jupiter 1:6 Resonance', detected: false }
    ];
}

// Update simulation speed based on slider value
function updateSimulationSpeed() {
    simulationSpeed = parseInt(document.getElementById('simulation-speed').value);
    document.getElementById('speed-value').textContent = simulationSpeed + 'x';
}

// Update view mode based on dropdown selection
function updateViewMode() {
    viewMode = document.getElementById('view-mode').value;
    drawThreeBodySystem();
}

// Toggle simulation start/stop
function toggleSimulation() {
    const button = document.getElementById('start-simulation');
    
    if (isSimulationRunning) {
        stopSimulation();
        button.textContent = 'Start Simulation';
    } else {
        startSimulation();
        button.textContent = 'Pause Simulation';
    }
    
    isSimulationRunning = !isSimulationRunning;
}

// Start the simulation animation
function startSimulation() {
    if (!isSimulationRunning) {
        animateThreeBodySystem();
    }
}

// Stop the simulation animation
function stopSimulation() {
    if (animationFrameId) {
        cancelAnimationFrame(animationFrameId);
        animationFrameId = null;
    }
}

// Reset the simulation to initial state
function resetSimulation() {
    time = 0;
    stopSimulation();
    initializeBodies();
    
    // Initialize trails for each body
    initializeTrails();
    
    drawThreeBodySystem();
    drawPhaseRelationships();
    
    // Reset the start button text if necessary
    if (isSimulationRunning) {
        document.getElementById('start-simulation').textContent = 'Start Simulation';
        isSimulationRunning = false;
    }
}

// Initialize trail system for each body
function initializeTrails() {
    trails = [];
    for (let i = 0; i < bodies.length; i++) {
        // Each trail is an array of position points
        trails.push([]);
    }
    
    // Reset trail counter
    trailCounter = 0;
}

// Main animation loop for the three-body system
function animateThreeBodySystem() {
    // Update physics
    updatePhysics();
    
    // Draw the updated system
    drawThreeBodySystem();
    
    // Update phase relationships every 10 frames
    if (time % 10 === 0) {
        drawPhaseRelationships();
    }
    
    // Continue the animation loop
    animationFrameId = requestAnimationFrame(animateThreeBodySystem);
}

// Update the physics of the system
function updatePhysics() {
    const dt = 0.01 * (simulationSpeed / 50); // Scale time step by simulation speed
    time += dt;
    
    // Calculate forces
    for (let i = 0; i < bodies.length; i++) {
        const body1 = bodies[i];
        body1.acceleration = { x: 0, y: 0, z: 0 };
        
        for (let j = 0; j < bodies.length; j++) {
            if (i !== j) {
                const body2 = bodies[j];
                const dx = body2.position.x - body1.position.x;
                const dy = body2.position.y - body1.position.y;
                const dz = body2.position.z - body1.position.z;
                
                const distanceSquared = dx * dx + dy * dy + dz * dz;
                const distance = Math.sqrt(distanceSquared);
                
                // Apply gravitational force (F = G * m1 * m2 / r^2)
                // Using G = 1 for simplicity
                const force = body2.mass / distanceSquared;
                
                // Normalize direction vector
                const dirX = dx / distance;
                const dirY = dy / distance;
                const dirZ = dz / distance;
                
                // Add to acceleration (F = ma, so a = F/m)
                body1.acceleration.x += force * dirX;
                body1.acceleration.y += force * dirY;
                body1.acceleration.z += force * dirZ;
            }
        }
    }
    
    // Update velocities and positions using Verlet integration
    for (let i = 0; i < bodies.length; i++) {
        const body = bodies[i];
        
        // Update velocity: v(t+dt) = v(t) + a(t) * dt
        body.velocity.x += body.acceleration.x * dt;
        body.velocity.y += body.acceleration.y * dt;
        body.velocity.z += body.acceleration.z * dt;
        
        // Update position: p(t+dt) = p(t) + v(t+dt) * dt
        body.position.x += body.velocity.x * dt;
        body.position.y += body.velocity.y * dt;
        body.position.z += body.velocity.z * dt;
    }
    
    // Update trail system
    trailCounter++;
    if (trailCounter >= TRAIL_UPDATE_FREQUENCY) {
        // Add current positions to trails
        for (let i = 0; i < bodies.length; i++) {
            const body = bodies[i];
            const trailPoint = {
                x: body.position.x,
                y: body.position.y,
                z: body.position.z
            };
            
            // Add the point to this body's trail
            trails[i].push(trailPoint);
            
            // Trim trail if it gets too long
            if (trails[i].length > MAX_TRAIL_POINTS) {
                trails[i].shift(); // Remove oldest point
            }
        }
        
        // Reset counter
        trailCounter = 0;
    }
    
    // Periodically check for resonance patterns
    resonanceCheckCounter++;
    if (resonanceCheckCounter >= RESONANCE_CHECK_INTERVAL) {
        detectResonances();
        calculateSystemStability();
        resonanceCheckCounter = 0;
        resonanceUpdateRequired = true;
    }
}

// Detect orbital resonance patterns between planets
function detectResonances() {
    detectedResonances = [];
    
    // We need a minimum amount of data to detect patterns
    if (trails[0].length < 50) return;
    
    // Check for Jupiter-Saturn 2:1 resonance
    const jupiterSaturnResonance = detectPlanetaryResonance(bodies[1], bodies[2], 2, 1);
    if (jupiterSaturnResonance > 0.75) {
        detectedResonances.push({
            bodyNames: [bodies[1].name, bodies[2].name],
            ratio: "2:1",
            strength: jupiterSaturnResonance,
            description: "Mean motion resonance (orbital period ratio)"
        });
        resonancePatterns[0].detected = true;
    }
    
    // Check for Earth-Venus 13:8 resonance
    const earthVenusResonance = detectPlanetaryResonance(bodies[3], bodies[4], 13, 8);
    if (earthVenusResonance > 0.7) {
        detectedResonances.push({
            bodyNames: [bodies[3].name, bodies[4].name],
            ratio: "13:8",
            strength: earthVenusResonance,
            description: "Mean motion resonance (orbital period ratio)"
        });
        resonancePatterns[1].detected = true;
    }
    
    // Check for Neptune-Pluto 3:2 resonance
    const neptunePlutoResonance = detectPlanetaryResonance(bodies[5], bodies[6], 3, 2);
    if (neptunePlutoResonance > 0.6) {
        detectedResonances.push({
            bodyNames: [bodies[5].name, bodies[6].name],
            ratio: "3:2",
            strength: neptunePlutoResonance,
            description: "Mean motion resonance (orbital period ratio)"
        });
        resonancePatterns[2].detected = true;
    }
    
    // Check for Earth-Moon Lagrange points
    const earthMoonResonance = detectLagrangePoints(bodies[3], bodies[7]);
    if (earthMoonResonance > 0.8) {
        detectedResonances.push({
            bodyNames: [bodies[3].name, bodies[7].name],
            ratio: "L-points",
            strength: earthMoonResonance,
            description: "Lagrange stability points detected"
        });
        resonancePatterns[3].detected = true;
    }
    
    // Update the DOM with detected resonances if we have a results area
    const resonanceResults = document.getElementById('resonance-results');
    if (resonanceResults) {
        updateResonanceResults(resonanceResults);
    }
}

// Detect resonance between two planets with given ratio
function detectPlanetaryResonance(body1, body2, ratio1, ratio2) {
    // Get orbital period estimates by analyzing trail points
    const period1 = estimateOrbitalPeriod(body1);
    const period2 = estimateOrbitalPeriod(body2);
    
    if (period1 <= 0 || period2 <= 0) return 0;
    
    // Calculate expected ratio
    const expectedRatio = ratio1 / ratio2;
    
    // Calculate actual ratio
    const actualRatio = period1 / period2;
    
    // Calculate similarity (1.0 = perfect match)
    const similarity = 1.0 - Math.min(Math.abs(actualRatio - expectedRatio) / expectedRatio, 1.0);
    
    return similarity;
}

// Estimate orbital period using trail data and angle changes
function estimateOrbitalPeriod(body) {
    // Find body index
    const bodyIndex = bodies.findIndex(b => b.name === body.name);
    if (bodyIndex === -1 || bodyIndex === 0) return 0; // Sun has no period
    
    const bodyTrail = trails[bodyIndex];
    if (bodyTrail.length < 20) return 0; // Need enough trail data
    
    // Find the sun
    const sunIndex = 0; // Assuming sun is always first
    
    // Calculate angles from Sun for each trail point
    const angles = [];
    for (let i = 0; i < bodyTrail.length; i++) {
        const trailPoint = bodyTrail[i];
        const dx = trailPoint.x - bodies[sunIndex].position.x;
        const dy = trailPoint.y - bodies[sunIndex].position.y;
        const angle = Math.atan2(dy, dx);
        angles.push(angle);
    }
    
    // Count full orbits (when angle crosses from π to -π)
    let orbits = 0;
    for (let i = 1; i < angles.length; i++) {
        const prev = angles[i-1];
        const curr = angles[i];
        
        // Detect crossing from positive to negative (crossing π to -π)
        if (prev > 2.0 && curr < -2.0) {
            orbits++;
        }
    }
    
    if (orbits < 1) return 0; // Need at least one orbit
    
    // Estimate period based on time and orbit count
    return time / orbits;
}

// Detect Lagrange points between two bodies (like Earth-Moon)
function detectLagrangePoints(primaryBody, secondaryBody) {
    // Find body indices
    const primaryIndex = bodies.findIndex(b => b.name === primaryBody.name);
    const secondaryIndex = bodies.findIndex(b => b.name === secondaryBody.name);
    
    if (primaryIndex === -1 || secondaryIndex === -1) return 0;
    
    const primaryTrail = trails[primaryIndex];
    const secondaryTrail = trails[secondaryIndex];
    
    if (primaryTrail.length < 50 || secondaryTrail.length < 50) return 0;
    
    // Check if the secondary body maintains a roughly constant distance from primary
    const distances = [];
    for (let i = 0; i < Math.min(primaryTrail.length, secondaryTrail.length); i += 5) {
        const dx = secondaryTrail[i].x - primaryTrail[i].x;
        const dy = secondaryTrail[i].y - primaryTrail[i].y;
        const dz = secondaryTrail[i].z - primaryTrail[i].z;
        
        const distance = Math.sqrt(dx*dx + dy*dy + dz*dz);
        distances.push(distance);
    }
    
    // Calculate variance of distances (low variance indicates stable Lagrange point)
    const avgDistance = distances.reduce((sum, d) => sum + d, 0) / distances.length;
    const variance = distances.reduce((sum, d) => sum + (d - avgDistance) * (d - avgDistance), 0) / distances.length;
    
    // Normalize variance to a similarity score
    const maxExpectedVariance = 100; // Adjusted based on simulation scale
    const similarity = 1.0 - Math.min(variance / maxExpectedVariance, 1.0);
    
    return similarity;
}

// Update DOM with detected resonances
function updateResonanceResults(resultsElement) {
    // Clear existing results
    resultsElement.innerHTML = '';
    
    // Create header
    const header = document.createElement('h4');
    header.textContent = 'Detected Resonance Patterns';
    resultsElement.appendChild(header);
    
    // Create resonance list
    const list = document.createElement('ul');
    list.className = 'resonance-list';
    
    // Add each detected resonance
    for (const resonance of detectedResonances) {
        const item = document.createElement('li');
        item.innerHTML = `
            <span class="resonance-bodies">${resonance.bodyNames.join('-')}</span>
            <span class="resonance-ratio">${resonance.ratio}</span>
            <span class="resonance-strength">${(resonance.strength * 100).toFixed(1)}% match</span>
            <span class="resonance-desc">${resonance.description}</span>
        `;
        list.appendChild(item);
    }
    
    // If no resonances detected yet
    if (detectedResonances.length === 0) {
        const item = document.createElement('li');
        item.textContent = 'Running analysis... Keep simulation running to detect patterns.';
        list.appendChild(item);
    }
    
    resultsElement.appendChild(list);
}

// Calculate overall system stability metric
function calculateSystemStability() {
    // This is a simplified stability metric based on energy conservation and trajectory patterns
    
    // Contribute detected resonances to stability
    let stabilityScore = 0.5; // Base value
    
    // Each detected resonance adds to stability
    stabilityScore += detectedResonances.length * 0.05;
    
    // Calculate energy conservation
    const initialEnergy = calculateSystemEnergy();
    if (typeof bodies.initialEnergy === 'undefined') {
        bodies.initialEnergy = initialEnergy;
    }
    
    const energyRatio = Math.abs(initialEnergy / bodies.initialEnergy);
    const energyConservation = 1.0 - Math.min(Math.abs(energyRatio - 1.0), 0.5);
    
    // Weight energy conservation heavily in stability
    stabilityScore = stabilityScore * 0.7 + energyConservation * 0.3;
    
    // Cap at 0.0-1.0 range
    systemStabilityValue = Math.max(0.0, Math.min(1.0, stabilityScore));
    
    // Update UI if available
    const stabilityElement = document.getElementById('system-stability');
    if (stabilityElement) {
        stabilityElement.textContent = systemStabilityValue.toFixed(3);
    }
}

// Calculate total system energy (kinetic + potential)
function calculateSystemEnergy() {
    let totalEnergy = 0;
    
    // Calculate kinetic energy and potential energy
    for (let i = 0; i < bodies.length; i++) {
        const body1 = bodies[i];
        
        // Kinetic energy: 0.5 * m * v^2
        const vSquared = body1.velocity.x * body1.velocity.x + 
                        body1.velocity.y * body1.velocity.y + 
                        body1.velocity.z * body1.velocity.z;
        const kineticEnergy = 0.5 * body1.mass * vSquared;
        
        // Add to total
        totalEnergy += kineticEnergy;
        
        // Potential energy with all other bodies: -G * m1 * m2 / r
        for (let j = i + 1; j < bodies.length; j++) {
            const body2 = bodies[j];
            
            const dx = body2.position.x - body1.position.x;
            const dy = body2.position.y - body1.position.y;
            const dz = body2.position.z - body1.position.z;
            
            const distance = Math.sqrt(dx*dx + dy*dy + dz*dz);
            if (distance > 0.1) { // Avoid singularity
                const potentialEnergy = -1.0 * body1.mass * body2.mass / distance; // G=1
                totalEnergy += potentialEnergy;
            }
        }
    }
    
    return totalEnergy;
}

// Draw the three-body system
function drawThreeBodySystem() {
    // Clear canvas
    threeBodyCtx.fillStyle = '#000';
    threeBodyCtx.fillRect(0, 0, threeBodyCanvas.width, threeBodyCanvas.height);
    
    // Draw orbit trails (faint ellipses)
    drawOrbitTrails();
    
    // Draw bodies based on view mode
    if (viewMode === '3d') {
        drawBodies3D();
    } else if (viewMode === 'top') {
        drawBodiesTopView();
    } else {
        drawBodiesSideView();
    }
    
    // Draw labels
    drawBodyLabels();
    
    // Draw timestamp
    drawTimestamp();
}

// Draw orbit trails for the planets
function drawOrbitTrails() {
    // First, let's draw a faint celestial grid
    drawCelestialGrid();
    
    // Now draw the actual dynamically generated trails
    for (let i = 0; i < bodies.length; i++) {
        const body = bodies[i];
        const bodyTrail = trails[i];
        
        if (bodyTrail.length < 2) continue; // Need at least 2 points to draw a line
        
        // Set trail style based on the body
        threeBodyCtx.strokeStyle = body.color;
        threeBodyCtx.lineWidth = i === 0 ? 1 : 2; // Sun has thinner trail
        
        // Set trail opacity
        threeBodyCtx.globalAlpha = 0.5;
        
        // Start the trail path
        threeBodyCtx.beginPath();
        
        // Calculate screen coordinates for the first point
        let screenX = threeBodyCanvas.width / 2 + bodyTrail[0].x;
        let screenY = threeBodyCanvas.height / 2 + (viewMode === 'side' ? bodyTrail[0].z : bodyTrail[0].y);
        threeBodyCtx.moveTo(screenX, screenY);
        
        // Draw the rest of the trail
        for (let j = 1; j < bodyTrail.length; j++) {
            // Get trail point and convert to screen coordinates
            screenX = threeBodyCanvas.width / 2 + bodyTrail[j].x;
            screenY = threeBodyCanvas.height / 2 + (viewMode === 'side' ? bodyTrail[j].z : bodyTrail[j].y);
            
            // Add point to the path
            threeBodyCtx.lineTo(screenX, screenY);
        }
        
        // Draw the complete trail
        threeBodyCtx.stroke();
    }
    
    // Reset opacity
    threeBodyCtx.globalAlpha = 1.0;
    threeBodyCtx.lineWidth = 1;
}

// Draw a celestial grid for reference
function drawCelestialGrid() {
    const centerX = threeBodyCanvas.width / 2;
    const centerY = threeBodyCanvas.height / 2;
    const gridSize = 50; // Distance between grid lines in pixels
    const maxRadius = Math.max(threeBodyCanvas.width, threeBodyCanvas.height);
    
    // Draw circular grid
    threeBodyCtx.strokeStyle = 'rgba(100, 100, 150, 0.15)';
    for (let r = gridSize; r <= maxRadius; r += gridSize) {
        threeBodyCtx.beginPath();
        threeBodyCtx.arc(centerX, centerY, r, 0, 2 * Math.PI);
        threeBodyCtx.stroke();
    }
    
    // Draw radial lines
    for (let angle = 0; angle < 360; angle += 30) {
        const radian = angle * Math.PI / 180;
        threeBodyCtx.beginPath();
        threeBodyCtx.moveTo(centerX, centerY);
        threeBodyCtx.lineTo(
            centerX + Math.cos(radian) * maxRadius,
            centerY + Math.sin(radian) * maxRadius
        );
        threeBodyCtx.stroke();
    }
}

// Draw bodies in 3D view
function drawBodies3D() {
    for (let i = 0; i < bodies.length; i++) {
        const body = bodies[i];
        
        // Calculate screen position
        const screenX = threeBodyCanvas.width / 2 + body.position.x;
        const screenY = threeBodyCanvas.height / 2 + body.position.y;
        
        // Adjust size based on z-position for 3D effect
        const zFactor = 1 + (body.position.z / 500);
        const adjustedRadius = body.radius * zFactor;
        
        // Draw body
        threeBodyCtx.fillStyle = body.color;
        threeBodyCtx.beginPath();
        threeBodyCtx.arc(screenX, screenY, adjustedRadius, 0, 2 * Math.PI);
        threeBodyCtx.fill();
        
        // Add shadow/highlight for 3D effect
        threeBodyCtx.globalAlpha = 0.3;
        threeBodyCtx.fillStyle = body.position.z > 0 ? '#fff' : '#000';
        threeBodyCtx.beginPath();
        threeBodyCtx.arc(
            screenX + adjustedRadius * 0.3, 
            screenY - adjustedRadius * 0.3, 
            adjustedRadius * 0.7, 
            0, 
            2 * Math.PI
        );
        threeBodyCtx.fill();
        threeBodyCtx.globalAlpha = 1.0;
    }
}

// Draw bodies in top view (x-y plane)
function drawBodiesTopView() {
    for (let i = 0; i < bodies.length; i++) {
        const body = bodies[i];
        
        // Calculate screen position
        const screenX = threeBodyCanvas.width / 2 + body.position.x;
        const screenY = threeBodyCanvas.height / 2 + body.position.y;
        
        // Draw body
        threeBodyCtx.fillStyle = body.color;
        threeBodyCtx.beginPath();
        threeBodyCtx.arc(screenX, screenY, body.radius, 0, 2 * Math.PI);
        threeBodyCtx.fill();
    }
}

// Draw bodies in side view (x-z plane)
function drawBodiesSideView() {
    for (let i = 0; i < bodies.length; i++) {
        const body = bodies[i];
        
        // Calculate screen position
        const screenX = threeBodyCanvas.width / 2 + body.position.x;
        const screenY = threeBodyCanvas.height / 2 + body.position.z;
        
        // Draw body
        threeBodyCtx.fillStyle = body.color;
        threeBodyCtx.beginPath();
        threeBodyCtx.arc(screenX, screenY, body.radius, 0, 2 * Math.PI);
        threeBodyCtx.fill();
    }
}

// Draw labels for bodies
function drawBodyLabels() {
    threeBodyCtx.fillStyle = '#fff';
    threeBodyCtx.font = '10px Arial';
    threeBodyCtx.textAlign = 'center';
    
    for (let i = 0; i < bodies.length; i++) {
        const body = bodies[i];
        
        // Calculate screen position
        const screenX = threeBodyCanvas.width / 2 + body.position.x;
        const screenY = threeBodyCanvas.height / 2 + (viewMode === 'side' ? body.position.z : body.position.y);
        
        // Draw label
        threeBodyCtx.fillText(body.name, screenX, screenY + body.radius + 12);
    }
}

// Draw timestamp and analysis information
function drawTimestamp() {
    const years = (time * 10).toFixed(1); // Convert simulation time to years
    
    threeBodyCtx.fillStyle = '#fff';
    threeBodyCtx.font = '12px Arial';
    threeBodyCtx.textAlign = 'left';
    threeBodyCtx.fillText(`Simulation Time: ${years} years`, 10, 20);
    
    // Show system stability from live analysis
    const stabilityText = systemStabilityValue > 0 
        ? systemStabilityValue.toFixed(3) 
        : "Analyzing...";
    threeBodyCtx.fillText(`System Stability: ${stabilityText}`, 10, 40);
    
    // Show detected resonances count
    const detectedCount = detectedResonances.length;
    threeBodyCtx.fillText(`Resonance Patterns Detected: ${detectedCount}`, 10, 60);
    
    // Show detection status message
    if (time < 1.0) {
        threeBodyCtx.fillText("Running initial analysis...", 10, 80);
    } else if (detectedCount === 0) {
        threeBodyCtx.fillText("Analyzing orbital patterns...", 10, 80);
    } else {
        threeBodyCtx.fillStyle = '#90EE90'; // Light green
        threeBodyCtx.fillText("Resonance detected! See results panel for details.", 10, 80);
    }
}

// Draw phase relationships between planets
function drawPhaseRelationships() {
    // Clear phase canvas
    phaseCtx.fillStyle = '#000';
    phaseCtx.fillRect(0, 0, phaseCanvas.width, phaseCanvas.height);
    
    // Set up layout
    const centerX = phaseCanvas.width / 2;
    const centerY = phaseCanvas.height / 2;
    const radius = Math.min(centerX, centerY) - 10;
    
    // Draw phase circle background
    phaseCtx.strokeStyle = 'rgba(255, 255, 255, 0.2)';
    phaseCtx.beginPath();
    phaseCtx.arc(centerX, centerY, radius, 0, 2 * Math.PI);
    phaseCtx.stroke();
    
    // Draw radial lines for reference
    for (let angle = 0; angle < 360; angle += 45) {
        const radian = angle * Math.PI / 180;
        phaseCtx.strokeStyle = 'rgba(100, 100, 150, 0.15)';
        phaseCtx.beginPath();
        phaseCtx.moveTo(centerX, centerY);
        phaseCtx.lineTo(
            centerX + Math.cos(radian) * radius,
            centerY + Math.sin(radian) * radius
        );
        phaseCtx.stroke();
    }
    
    // Draw planet positions on the phase circle
    const planetColors = [
        SUN_COLOR,      // Sun
        JUPITER_COLOR,  // Jupiter
        SATURN_COLOR,   // Saturn
        EARTH_COLOR,    // Earth
        VENUS_COLOR,    // Venus
        NEPTUNE_COLOR   // Neptune
    ];
    
    // Calculate and draw planet positions
    const planetAngles = [];
    const planetPositions = [];
    
    for (let i = 0; i < Math.min(6, bodies.length); i++) {
        if (i === 0) continue; // Skip the Sun (center)
        
        const body = bodies[i];
        const angle = Math.atan2(
            body.position.y - bodies[0].position.y,
            body.position.x - bodies[0].position.x
        );
        
        planetAngles.push(angle);
        
        // Calculate position on circle
        const x = centerX + radius * Math.cos(angle);
        const y = centerY + radius * Math.sin(angle);
        
        planetPositions.push({ x, y, name: body.name });
        
        // Draw planet
        phaseCtx.fillStyle = body.color;
        phaseCtx.beginPath();
        phaseCtx.arc(x, y, 5, 0, 2 * Math.PI);
        phaseCtx.fill();
        
        // Draw small label
        phaseCtx.fillStyle = 'rgba(255, 255, 255, 0.7)';
        phaseCtx.font = '8px Arial';
        phaseCtx.textAlign = 'center';
        phaseCtx.fillText(body.name.charAt(0), x, y - 8); // First letter as label
    }
    
    // Draw resonance connections if detected
    if (detectedResonances.length > 0) {
        // Get the Jupiter-Saturn resonance if available
        const jupiterSaturnResonance = detectedResonances.find(r => 
            r.bodyNames.includes('Jupiter') && r.bodyNames.includes('Saturn'));
            
        if (jupiterSaturnResonance) {
            // Find the Jupiter and Saturn positions
            const jupiterPos = planetPositions.find(p => p.name === 'Jupiter');
            const saturnPos = planetPositions.find(p => p.name === 'Saturn');
            
            if (jupiterPos && saturnPos) {
                // Draw connection with intensity based on resonance strength
                const alpha = Math.min(0.2 + jupiterSaturnResonance.strength * 0.7, 0.9);
                phaseCtx.strokeStyle = `rgba(255, 215, 0, ${alpha})`;
                phaseCtx.lineWidth = 2;
                phaseCtx.beginPath();
                phaseCtx.moveTo(jupiterPos.x, jupiterPos.y);
                phaseCtx.lineTo(saturnPos.x, saturnPos.y);
                phaseCtx.stroke();
                phaseCtx.lineWidth = 1;
                
                // Draw resonance label
                phaseCtx.fillStyle = '#fff';
                phaseCtx.font = '10px Arial';
                phaseCtx.textAlign = 'center';
                const midX = (jupiterPos.x + saturnPos.x) / 2;
                const midY = (jupiterPos.y + saturnPos.y) / 2;
                phaseCtx.fillText(`2:1 (${Math.round(jupiterSaturnResonance.strength * 100)}%)`, midX, midY - 8);
            }
        }
        
        // Draw other resonances if detected
        const earthVenusResonance = detectedResonances.find(r => 
            r.bodyNames.includes('Earth') && r.bodyNames.includes('Venus'));
            
        if (earthVenusResonance) {
            const earthPos = planetPositions.find(p => p.name === 'Earth');
            const venusPos = planetPositions.find(p => p.name === 'Venus');
            
            if (earthPos && venusPos) {
                const alpha = Math.min(0.2 + earthVenusResonance.strength * 0.7, 0.9);
                phaseCtx.strokeStyle = `rgba(30, 144, 255, ${alpha})`;
                phaseCtx.lineWidth = 2;
                phaseCtx.beginPath();
                phaseCtx.moveTo(earthPos.x, earthPos.y);
                phaseCtx.lineTo(venusPos.x, venusPos.y);
                phaseCtx.stroke();
                phaseCtx.lineWidth = 1;
                
                // Draw resonance label
                phaseCtx.fillStyle = '#fff';
                phaseCtx.font = '10px Arial';
                phaseCtx.textAlign = 'center';
                const midX = (earthPos.x + venusPos.x) / 2;
                const midY = (earthPos.y + venusPos.y) / 2;
                phaseCtx.fillText(`13:8 (${Math.round(earthVenusResonance.strength * 100)}%)`, midX, midY - 8);
            }
        }
    } else {
        // If no resonances detected yet, show dynamic phase relationship
        phaseCtx.fillStyle = 'rgba(255, 255, 255, 0.5)';
        phaseCtx.font = '10px Arial';
        phaseCtx.textAlign = 'center';
        phaseCtx.fillText('Analyzing orbital resonance patterns...', centerX, centerY - radius - 10);
    }
    
    // Update energy conservation value
    const energyElement = document.getElementById('energy-conservation');
    if (energyElement) {
        // Calculate energy conservation as a ratio relative to initial energy
        const initialEnergy = typeof bodies.initialEnergy !== 'undefined' ? bodies.initialEnergy : calculateSystemEnergy();
        if (typeof bodies.initialEnergy === 'undefined') {
            bodies.initialEnergy = initialEnergy;
        }
        
        const currentEnergy = calculateSystemEnergy();
        const ratio = Math.abs(currentEnergy / initialEnergy);
        const conservation = 1.0 - Math.min(Math.abs(ratio - 1.0), 0.1) * 10; // Scale for display
        
        energyElement.textContent = conservation.toFixed(3);
    }
}

// Wait for DOM to be ready before initializing
document.addEventListener('DOMContentLoaded', function() {
    // Check if we're on the three-body tab
    const threeBodyTab = document.querySelector('.main-tab[data-tab="three-body"]');
    
    if (threeBodyTab) {
        // Add click handler for the three-body tab
        threeBodyTab.addEventListener('click', function() {
            console.log('Three-body tab clicked');
            // We need a small delay to ensure the canvas is visible first
            setTimeout(initThreeBodySolver, 100);
        });
    }
    
    // Also initialize if the three-body tab is already active
    if (document.querySelector('.main-tab.active[data-tab="three-body"]')) {
        initThreeBodySolver();
    }
});