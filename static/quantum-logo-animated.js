// Animated quantum logo with oscillating waves and quantum-inspired effects
document.addEventListener('DOMContentLoaded', function() {
  // Get the logo container
  const logoContainer = document.querySelector('.quantum-logo');
  
  if (!logoContainer) return;
  
  // Create canvas element
  const canvas = document.createElement('canvas');
  canvas.width = 180;
  canvas.height = 180;
  logoContainer.appendChild(canvas);
  
  const ctx = canvas.getContext('2d');
  
  // Animation variables
  let time = 0;
  const centerX = canvas.width / 2;
  const centerY = canvas.height / 2;
  
  // Animation function
  function animate() {
    // Clear canvas
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    
    // Create background glow
    const bgGlow = ctx.createRadialGradient(
      centerX, centerY, 10,
      centerX, centerY, 90
    );
    bgGlow.addColorStop(0, 'rgba(0, 183, 255, 0.05)');
    bgGlow.addColorStop(1, 'rgba(0, 183, 255, 0)');
    
    ctx.beginPath();
    ctx.arc(centerX, centerY, 90, 0, Math.PI * 2);
    ctx.fillStyle = bgGlow;
    ctx.fill();
    
    // Draw multiple orbital rings with different opacities
    const ringCount = 3;
    for (let i = 0; i < ringCount; i++) {
      const ringRadius = 40 + (i * 15);
      const ringOpacity = 0.3 - (i * 0.08);
      
      // Slightly shift the orbit for a more dynamic effect
      const offsetX = Math.sin(time * 0.01) * 2;
      const offsetY = Math.cos(time * 0.015) * 2;
      
      ctx.beginPath();
      ctx.ellipse(
        centerX + offsetX, 
        centerY + offsetY, 
        ringRadius, 
        ringRadius * 0.6, 
        time * 0.005, 
        0, 
        Math.PI * 2
      );
      ctx.strokeStyle = `rgba(0, 183, 255, ${ringOpacity})`;
      ctx.lineWidth = 1;
      ctx.stroke();
    }
    
    // Draw pulsing inner core
    const pulseSize = 5 * Math.sin(time * 0.05) + 20;
    ctx.beginPath();
    ctx.arc(centerX, centerY, pulseSize, 0, Math.PI * 2);
    ctx.fillStyle = '#00b7ff';
    ctx.fill();
    
    // Add core glow effect
    const coreGlow = ctx.createRadialGradient(
      centerX, centerY, pulseSize * 0.5,
      centerX, centerY, pulseSize * 2
    );
    coreGlow.addColorStop(0, 'rgba(0, 183, 255, 0.5)');
    coreGlow.addColorStop(1, 'rgba(0, 183, 255, 0)');
    
    ctx.beginPath();
    ctx.arc(centerX, centerY, pulseSize * 2, 0, Math.PI * 2);
    ctx.fillStyle = coreGlow;
    ctx.fill();
    
    // Draw quantum particles in orbital paths
    const particleCount = 5;
    for (let i = 0; i < particleCount; i++) {
      // Each particle has its own orbital speed and path
      const orbitSpeed = 0.02 + (i * 0.005);
      const orbitRadius = 35 + (i * 8);
      const orbitEccentricity = 0.8 - (i * 0.1);
      
      const angle = (time * orbitSpeed) + (i * (Math.PI * 2 / particleCount));
      const x = centerX + Math.cos(angle) * orbitRadius * orbitEccentricity;
      const y = centerY + Math.sin(angle) * orbitRadius;
      
      // Particle size pulsates slightly
      const particleSize = 3 + Math.sin(time * 0.1 + i) * 1;
      
      ctx.beginPath();
      ctx.arc(x, y, particleSize, 0, Math.PI * 2);
      ctx.fillStyle = '#00b7ff';
      ctx.fill();
      
      // Particle glow effect
      const particleGlow = ctx.createRadialGradient(
        x, y, 1, x, y, 8
      );
      particleGlow.addColorStop(0, 'rgba(0, 183, 255, 0.4)');
      particleGlow.addColorStop(1, 'rgba(0, 183, 255, 0)');
      
      ctx.beginPath();
      ctx.arc(x, y, 8, 0, Math.PI * 2);
      ctx.fillStyle = particleGlow;
      ctx.fill();
      
      // Occasionally draw 'quantum entanglement' lines between particles
      if (i > 0 && Math.sin(time * 0.02 + i) > 0.7) {
        const prevI = (i - 1) % particleCount;
        const prevAngle = (time * (0.02 + (prevI * 0.005))) + (prevI * (Math.PI * 2 / particleCount));
        const prevX = centerX + Math.cos(prevAngle) * (35 + (prevI * 8)) * (0.8 - (prevI * 0.1));
        const prevY = centerY + Math.sin(prevAngle) * (35 + (prevI * 8));
        
        ctx.beginPath();
        ctx.moveTo(x, y);
        ctx.lineTo(prevX, prevY);
        ctx.strokeStyle = 'rgba(0, 183, 255, 0.2)';
        ctx.lineWidth = 1;
        ctx.stroke();
      }
    }
    
    // Draw multiple wave oscillations
    for (let w = 0; w < 2; w++) {
      ctx.beginPath();
      const waveAmplitude = 8 + (w * 5);
      const waveFrequency = 0.15 + (w * 0.05);
      const waveSpeed = 0.04 + (w * 0.01);
      const waveY = centerY + 10 + (w * 20);
      
      ctx.moveTo(centerX - 70, waveY);
      
      for (let x = -70; x <= 70; x++) {
        const y = Math.sin((x * waveFrequency) + (time * waveSpeed)) * waveAmplitude;
        ctx.lineTo(centerX + x, waveY + y);
      }
      
      ctx.strokeStyle = `rgba(0, 183, 255, ${0.7 - (w * 0.3)})`;
      ctx.lineWidth = 2 - (w * 0.5);
      ctx.stroke();
    }
    
    // Update time and request next frame
    time++;
    requestAnimationFrame(animate);
  }
  
  // Start animation
  animate();
});