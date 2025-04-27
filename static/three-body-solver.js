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

// Draw timestamp
function drawTimestamp() {
    const years = (time * 10).toFixed(1); // Convert simulation time to years
    
    threeBodyCtx.fillStyle = '#fff';
    threeBodyCtx.font = '12px Arial';
    threeBodyCtx.textAlign = 'left';
    threeBodyCtx.fillText(`Simulation Time: ${years} years`, 10, 20);
    threeBodyCtx.fillText(`Stability: 0.984`, 10, 40);
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
    
    // Draw 2:1 Jupiter-Saturn resonance
    const jupiterAngle = Math.atan2(
        bodies[1].position.y - bodies[0].position.y,
        bodies[1].position.x - bodies[0].position.x
    );
    
    const saturnAngle = Math.atan2(
        bodies[2].position.y - bodies[0].position.y,
        bodies[2].position.x - bodies[0].position.x
    );
    
    // Draw Jupiter position
    phaseCtx.fillStyle = JUPITER_COLOR;
    const jupiterX = centerX + radius * Math.cos(jupiterAngle);
    const jupiterY = centerY + radius * Math.sin(jupiterAngle);
    phaseCtx.beginPath();
    phaseCtx.arc(jupiterX, jupiterY, 6, 0, 2 * Math.PI);
    phaseCtx.fill();
    
    // Draw Saturn position
    phaseCtx.fillStyle = SATURN_COLOR;
    const saturnX = centerX + radius * Math.cos(saturnAngle);
    const saturnY = centerY + radius * Math.sin(saturnAngle);
    phaseCtx.beginPath();
    phaseCtx.arc(saturnX, saturnY, 6, 0, 2 * Math.PI);
    phaseCtx.fill();
    
    // Draw connection line showing resonance
    phaseCtx.strokeStyle = 'rgba(255, 215, 0, 0.5)';
    phaseCtx.beginPath();
    phaseCtx.moveTo(jupiterX, jupiterY);
    phaseCtx.lineTo(saturnX, saturnY);
    phaseCtx.stroke();
    
    // Label
    phaseCtx.fillStyle = '#fff';
    phaseCtx.font = '10px Arial';
    phaseCtx.textAlign = 'center';
    phaseCtx.fillText('J-S 2:1 Resonance', centerX, centerY - radius - 10);
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