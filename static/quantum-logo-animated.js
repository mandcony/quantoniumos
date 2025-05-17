// Simple pulsating blue dot animation to match the reference image
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
    
    // Create a subtle outer ring
    ctx.beginPath();
    ctx.arc(centerX, centerY, 60, 0, Math.PI * 2);
    ctx.strokeStyle = 'rgba(0, 183, 255, 0.3)';
    ctx.lineWidth = 1;
    ctx.stroke();
    
    // Simple blue dot with subtle pulsing
    const dotSize = 10;
    ctx.beginPath();
    ctx.arc(centerX, centerY, dotSize, 0, Math.PI * 2);
    ctx.fillStyle = '#00b7ff';
    ctx.fill();
    
    // Add glow effect around the dot
    const glow = ctx.createRadialGradient(
      centerX, centerY, dotSize * 0.5,
      centerX, centerY, dotSize * 2
    );
    glow.addColorStop(0, 'rgba(0, 183, 255, 0.6)');
    glow.addColorStop(1, 'rgba(0, 183, 255, 0)');
    
    ctx.beginPath();
    ctx.arc(centerX, centerY, dotSize * 2, 0, Math.PI * 2);
    ctx.fillStyle = glow;
    ctx.fill();
    
    // Update time and request next frame
    time++;
    requestAnimationFrame(animate);
  }
  
  // Start animation
  animate();
});