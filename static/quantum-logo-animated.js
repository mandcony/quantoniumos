// Animated quantum logo with oscillating waves
document.addEventListener('DOMContentLoaded', function() {
  // Get the logo container
  const logoContainer = document.querySelector('.quantum-logo');
  
  if (!logoContainer) return;
  
  // Create canvas element
  const canvas = document.createElement('canvas');
  canvas.width = 150;
  canvas.height = 150;
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
    
    // Draw outer circle
    ctx.beginPath();
    ctx.arc(centerX, centerY, 50, 0, Math.PI * 2);
    ctx.strokeStyle = 'rgba(0, 183, 255, 0.3)';
    ctx.lineWidth = 1;
    ctx.stroke();
    
    // Draw pulsing inner circle
    const pulseSize = 4 * Math.sin(time * 0.05) + 15;
    ctx.beginPath();
    ctx.arc(centerX, centerY, pulseSize, 0, Math.PI * 2);
    ctx.fillStyle = '#00b7ff';
    ctx.fill();
    
    // Add glow effect
    const glow = ctx.createRadialGradient(
      centerX, centerY, pulseSize * 0.8,
      centerX, centerY, pulseSize * 2
    );
    glow.addColorStop(0, 'rgba(0, 183, 255, 0.4)');
    glow.addColorStop(1, 'rgba(0, 183, 255, 0)');
    
    ctx.beginPath();
    ctx.arc(centerX, centerY, pulseSize * 2, 0, Math.PI * 2);
    ctx.fillStyle = glow;
    ctx.fill();
    
    // Draw orbiting particles
    for (let i = 0; i < 3; i++) {
      const angle = (time * 0.03) + (i * Math.PI * 2 / 3);
      const orbitRadius = 35;
      const x = centerX + Math.cos(angle) * orbitRadius;
      const y = centerY + Math.sin(angle) * orbitRadius;
      
      ctx.beginPath();
      ctx.arc(x, y, 3, 0, Math.PI * 2);
      ctx.fillStyle = '#00b7ff';
      ctx.fill();
      
      // Particle glow
      const particleGlow = ctx.createRadialGradient(
        x, y, 1, x, y, 6
      );
      particleGlow.addColorStop(0, 'rgba(0, 183, 255, 0.3)');
      particleGlow.addColorStop(1, 'rgba(0, 183, 255, 0)');
      
      ctx.beginPath();
      ctx.arc(x, y, 6, 0, Math.PI * 2);
      ctx.fillStyle = particleGlow;
      ctx.fill();
    }
    
    // Draw wave oscillation
    ctx.beginPath();
    const waveAmplitude = 10;
    const waveFrequency = 0.2;
    const waveSpeed = 0.05;
    
    ctx.moveTo(centerX - 50, centerY);
    
    for (let x = -50; x <= 50; x++) {
      const y = Math.sin((x * waveFrequency) + (time * waveSpeed)) * waveAmplitude;
      ctx.lineTo(centerX + x, centerY + y);
    }
    
    ctx.strokeStyle = 'rgba(0, 183, 255, 0.6)';
    ctx.lineWidth = 2;
    ctx.stroke();
    
    // Update time and request next frame
    time++;
    requestAnimationFrame(animate);
  }
  
  // Start animation
  animate();
});