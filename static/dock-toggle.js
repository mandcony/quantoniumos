// Dock toggle functionality
window.addEventListener('DOMContentLoaded', function() {
    const dockContainer = document.querySelector('.os-dock-container');
    const dockToggle = document.querySelector('.dock-toggle');
    const osContent = document.querySelector('.os-content');
    const osHeader = document.querySelector('.os-header');
    
    if (dockToggle && dockContainer) {
        console.log("Dock toggle initialized");
        
        dockToggle.addEventListener('click', function() {
            console.log("Dock toggle clicked");
            dockContainer.classList.toggle('hidden');
            
            // Adjust content margin when dock is hidden
            if (dockContainer.classList.contains('hidden')) {
                osContent.style.marginLeft = '20px';
                osHeader.style.marginLeft = '20px';
            } else {
                osContent.style.marginLeft = '80px';
                osHeader.style.marginLeft = '80px';
            }
        });
    } else {
        console.log("Dock toggle elements not found");
    }
});