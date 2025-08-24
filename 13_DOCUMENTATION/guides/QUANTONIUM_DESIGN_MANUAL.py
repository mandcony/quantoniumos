#!/usr/bin/env python3
"""
QuantoniumOS Design System and UI Documentation
===============================================
Comprehensive design guide for the QuantoniumOS user interface.
- Visual Language: Cream background, oval dock, black icons, Microsoft-blue buttons
- Design System: Centralized styling and proportions
- App Integration: Consistent UI for all applications
"""

# Design System Overview
# =====================
"""
QuantoniumOS Design System
--------------------------
The QuantoniumOS Design System provides a centralized approach to UI styling
with the following key characteristics:

1. Color Palette:
   - Primary Background: Cream (#f0ead6)
   - Secondary Background: Light Cream (#f8f6f0)
   - Text Primary: Dark Gray (#2d2d2d)
   - Text Secondary: Medium Gray (#555555)
   - Accent Blue: Microsoft Blue (#0078d4)
   - Logo Gray: Medium Gray (#6b6b6b)
   - Icon Black: Pure Black (#000000)

2. Typography:
   - Primary Font: Segoe UI
   - Monospace Font: Consolas
   - Dynamic sizing based on screen proportions

3. Dock Design:
   - Oval/D-shaped expandable tab on left side
   - Semi-transparent overlay when expanded
   - App icons arranged in semi-circular pattern when expanded

4. App Windows:
   - Consistent styling across all applications
   - Standard window size (70% of screen width, 80% of screen height)
   - Unified button styles, input fields, and UI elements

5. Proportions:
   - Golden ratio-based scaling
   - Responsive to different screen sizes
   - Calculated optimal spacing
"""

# UI Component Details
# ===================
"""
UI Components
------------
1. Dock Tab:
   - Location: Left side of screen
   - Default State: Collapsed (D-shape/oval)
   - Expanded State: Semi-circular app grid
   - Click to toggle expansion

2. App Icons:
   - Black silhouette icons using qtawesome
   - Circular background
   - Text labels beneath icons
   - Click to launch corresponding app

3. Q Logo:
   - Gray "Q" in center of screen
   - Large, bold font
   - Semi-transparent

4. Clock:
   - Top-right corner
   - Shows time and date
   - Bold font for readability

5. Buttons:
   - Microsoft blue (#0078d4)
   - White text
   - Rounded corners (4px radius)
   - Hover state slightly darker

6. Input Fields:
   - White background
   - Light gray border
   - Focus state with blue border

7. Container Elements:
   - Light cream background
   - Subtle borders
   - 8px corner radius
"""

# App Integration Guide
# ====================
"""
App Integration Guide
--------------------
1. Launcher Scripts:
   - Each app has a dedicated launcher script
   - Uses quantonium_app_wrapper.py for consistent styling
   - Example: launch_system_monitor.py

2. Design System Integration:
   - Import from quantonium_design_system.py
   - Use apply_unified_style() function
   - Reference design_system for proportions

3. App Development Guidelines:
   - Create core functionality as a PyQt widget
   - Let wrapper handle window management and styling
   - Focus on functionality, not visual design

4. Example Integration:
```python
#!/usr/bin/env python3
# launch_my_app.py
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from quantonium_app_wrapper import launch_app
from my_app import MyApp

def main():
    return launch_app(MyApp, "My App", "My Application")

if __name__ == "__main__":
    sys.exit(main())
```
"""

# Dock Implementation Details
# ==========================
"""
Dock Implementation Details
--------------------------
1. Oval/D-shaped Tab:
   - Left side of screen
   - Vertically centered
   - Collapsed state: Small oval tab
   - Expanded state: Semi-circular area showing apps

2. App Arrangement:
   - Semi-circular layout when dock expanded
   - 9 main apps arranged in an arc
   - Calculated positions using trigonometry
   - Consistent spacing between icons

3. Expansion Animation:
   - Smooth transition between states
   - App icons appear when expanded
   - App icons hide when collapsed

4. Implementation:
   - QGraphicsScene and QGraphicsView for rendering
   - QGraphicsEllipseItem for the dock shape
   - Design system calculates all proportions
   - App proxies positioned based on calculated coordinates
"""

# Global UI Elements
# =================
"""
Global UI Elements
----------------
1. Background:
   - Cream color (#f0ead6)
   - Consistent across all screens

2. Q Logo:
   - Center of screen when no apps are active
   - Gray color (#6b6b6b)
   - Large font size

3. System Clock:
   - Top-right corner
   - Shows time and date
   - Updates every second

4. System Tray:
   - Minimized state for running apps
   - Quick access to system functions
"""

# Conclusion
# ==========
"""
Implementation Notes
------------------
The QuantoniumOS user interface separates design from functionality through:

1. Centralized Design System:
   - All visual properties defined in quantonium_design_system.py
   - Global changes possible from a single location

2. App Wrapper:
   - quantonium_app_wrapper.py handles consistent styling
   - Apps focus only on functionality

3. Unified OS:
   - quantonium_os_unified.py implements the main OS interface
   - Uses design system for all visual elements

4. Launcher Scripts:
   - Individual launchers for each application
   - Consistent integration with the design system

This architecture allows for rapid UI changes without modifying underlying
application logic, fulfilling the separation of display from functionality.
"""
