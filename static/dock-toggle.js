// Dock toggle functionality
window.addEventListener('DOMContentLoaded', function() {
    const dockContainer = document.querySelector('.os-dock-container');
    const dockToggle = document.querySelector('.dock-toggle');
    const osContent = document.querySelector('.os-content');
    const osHeader = document.querySelector('.os-header');
    
    // Initialize event listener for the dock toggle button
    if (dockToggle && dockContainer) {
        console.log("Dock toggle initialized");
        
        // Function to toggle the dock visibility
        function toggleDock() {
            console.log("Dock toggle clicked");
            
            // Toggle the 'hidden' class on the dock container
            dockContainer.classList.toggle('hidden');
            
            // Update margins based on the dock state
            if (dockContainer.classList.contains('hidden')) {
                // When dock is hidden, reduce margins to give more space to content
                osContent.style.marginLeft = '20px';
                osHeader.style.marginLeft = '20px';
            } else {
                // When dock is visible, add margins to make room for dock
                osContent.style.marginLeft = '80px';
                osHeader.style.marginLeft = '80px';
            }
        }
        
        // Add click event listener to the toggle button
        dockToggle.addEventListener('click', toggleDock);
        
        // Add keyboard shortcut for toggling the dock (Alt+D)
        document.addEventListener('keydown', function(event) {
            if (event.altKey && event.key === 'd') {
                toggleDock();
            }
        });
        
        // Automatically toggle to make the initial state clear to the user
        // dockContainer.classList.add('hidden');
        // osContent.style.marginLeft = '20px';
        // osHeader.style.marginLeft = '20px';
    } else {
        console.error("Dock toggle elements not found");
    }
});