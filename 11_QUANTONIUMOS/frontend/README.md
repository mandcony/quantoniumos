# 🌌 QuantoniumOS Frontend Integration Guide

## 🚀 Quick Start

Welcome to the QuantoniumOS frontend system! This guide will help you get up and running with the complete quantum-inspired user interface.

### Installation

1. **Run the installer:**
   ```bash
   python frontend/scripts/install_quantonium.py
   ```

2. **Launch QuantoniumOS:**
   - **Windows:** Double-click the desktop shortcut or run `frontend/scripts/launch_quantonium.bat`
   - **VS Code:** Press `Ctrl+Shift+Q` or use Command Palette → "Launch QuantoniumOS"
   - **Manual:** `python frontend/ui/quantum_app_controller.py`

### 📁 Project Structure

```
frontend/
├── styles/                 # QSS stylesheets
│   └── quantonium_master.qss
├── ui/                     # Main UI controllers
│   └── quantum_app_controller.py
├── components/             # Reusable UI components
│   └── window_manager.py
├── extensions/             # VS Code integration
│   ├── quantonium_vscode.js
│   └── package.json
├── scripts/                # Installation & launch scripts
│   ├── install_quantonium.py
│   ├── launch_quantonium.ps1
│   └── launch_quantonium.bat
└── resources/              # Icons & assets
```

## 🎨 Architecture Overview

### Core Components

1. **Quantum App Controller** (`ui/quantum_app_controller.py`)
   - Main application window with tab system
   - Integration with all quantum applications
   - Menu bar, toolbar, and status monitoring

2. **Window Manager** (`components/window_manager.py`)
   - Advanced window arrangement (cascade, tile)
   - Quantum animations and effects
   - Session management
   - VS Code window detection

3. **QSS Master Stylesheet** (`styles/quantonium_master.qss`)
   - Quantum-inspired dark theme
   - Glassmorphism effects
   - Animated components
   - Responsive design

4. **VS Code Extension** (`extensions/`)
   - Command palette integration
   - Context menu commands
   - File analysis
   - Live status monitoring

### Application Flow

```
VS Code Extension ←→ Command Files ←→ QuantoniumOS
     ↓                    ↓                ↓
File Analysis      Window Management    App Controller
     ↓                    ↓                ↓
Quantum Detection  Session Handling   Tab Management
```

## 🔧 Configuration

### Main Config (`.quantonium/config.json`)

```json
{
  "version": "2.0.0",
  "frontend": {
    "theme": "quantum-dark",
    "window_management": true,
    "animations": true,
    "auto_save_session": true
  },
  "backend": {
    "auto_start": true,
    "quantum_validation": true,
    "crypto_validation": true
  },
  "vscode": {
    "integration": true,
    "auto_analyze": true,
    "status_bar": true
  }
}
```

### VS Code Settings

```json
{
  "quantonium.autoAnalyze": true,
  "quantonium.windowManagement": true,
  "quantonium.quantumValidation": true,
  "quantonium.theme": "quantum-dark",
  "quantonium.enableAnimations": true
}
```

## 🎮 Usage

### VS Code Integration

**Available Commands:**
- `Ctrl+Shift+Q` - Launch QuantoniumOS
- `Ctrl+Shift+R` - Launch RFT Visualizer
- `Ctrl+Shift+C` - Launch Quantum Crypto
- `Ctrl+Shift+W` - Cascade Windows
- `Ctrl+Shift+A` - Analyze Quantum Properties

**Context Menus:**
- Right-click Python files → "Analyze Quantum Properties"
- Right-click crypto files → "Validate Cryptography"

### Window Management

**Arrangement Options:**
- **Cascade:** Staggered window layout
- **Tile Horizontal:** Side-by-side arrangement
- **Tile Vertical:** Stacked arrangement

**Session Management:**
- Auto-save window positions and states
- Restore sessions on startup
- Manual save/load via menu

### Quick Launch Panel

**Available Applications:**
- 🔬 RFT Visualizer
- 🔐 Quantum Crypto
- 📊 System Monitor
- 🌌 Quantum Simulator
- 📝 Q-Notes
- 🌐 Q-Browser
- 🔒 Q-Vault
- 📧 Q-Mail

## 🎯 Key Features

### 1. Quantum-Inspired UI
- **Glassmorphism:** Translucent panels with blur effects
- **Quantum Glow:** Animated border effects
- **Pulse Animation:** Breathing button animations
- **Wave Motion:** Flowing background patterns

### 2. Advanced Window Management
- **Smart Arrangement:** Automatic window positioning
- **Animation System:** Smooth transitions and effects
- **Multi-Monitor:** Full multi-display support
- **Session Persistence:** Remember layouts between sessions

### 3. VS Code Deep Integration
- **Command Palette:** All functions accessible via Ctrl+Shift+P
- **File Analysis:** Automatic quantum content detection
- **Live Monitoring:** Real-time QuantoniumOS status
- **Context Actions:** Right-click menu integration

### 4. System Monitoring
- **Resource Tracking:** CPU, memory, and quantum state
- **Window Counting:** Active application tracking
- **Performance Optimization:** Efficient resource usage
- **Error Handling:** Graceful degradation

## 🔬 Development

### Adding New Applications

1. **Create App Module:**
   ```python
   # apps/my_quantum_app.py
   class MyQuantumApp(QWidget):
       def __init__(self):
           super().__init__()
           self.init_ui()
   ```

2. **Register in Controller:**
   ```python
   # Add to app_modules in quantum_app_controller.py
   'my_quantum_app': 'apps.my_quantum_app'
   ```

3. **Add Launch Button:**
   ```python
   # Add to apps list in create_left_panel()
   ("🔬 My Quantum App", "my_quantum_app")
   ```

### Custom Styling

**QSS Classes Available:**
- `.quantum-glow` - Glowing border effect
- `.quantum-pulse` - Breathing animation
- `.quantum-glass` - Glassmorphism panel
- `.quantum-button` - Standard button style
- `.quantum-frame` - Container frame style

**Example Custom Widget:**
```python
class MyWidget(QWidget):
    def __init__(self):
        super().__init__()
        self.setProperty("class", "quantum-glass")
        # Widget automatically gets quantum styling
```

### VS Code Extension Development

**Add New Command:**
```javascript
// In quantonium_vscode.js
const myCommand = vscode.commands.registerCommand('quantonium.my.command', () => {
    // Command implementation
});
context.subscriptions.push(myCommand);
```

**Update package.json:**
```json
{
  "contributes": {
    "commands": [{
      "command": "quantonium.my.command",
      "title": "My Quantum Command",
      "category": "QuantoniumOS"
    }]
  }
}
```

## 🐛 Troubleshooting

### Common Issues

**PyQt5 Import Error:**
```bash
pip install PyQt5
# or for conda:
conda install pyqt
```

**VS Code Extension Not Loading:**
1. Check if extension is installed: `code --list-extensions`
2. Reload VS Code window: `Ctrl+Shift+P` → "Reload Window"
3. Check extension logs: `Help` → `Toggle Developer Tools` → `Console`

**Window Manager Issues:**
1. Ensure PyWin32 is installed (Windows): `pip install pywin32`
2. Check window permissions
3. Verify VS Code is running

**Missing Stylesheet:**
1. Verify `frontend/styles/quantonium_master.qss` exists
2. Check file permissions
3. Restart QuantoniumOS

### Debug Mode

**Enable Debug Logging:**
```python
# Set environment variable before launching
import os
os.environ['QUANTONIUM_DEBUG'] = '1'
```

**VS Code Debug:**
```json
// In launch.json
{
  "name": "Debug QuantoniumOS",
  "type": "python",
  "request": "launch",
  "program": "${workspaceFolder}/frontend/ui/quantum_app_controller.py",
  "env": {"QUANTONIUM_DEBUG": "1"}
}
```

### Performance Optimization

**Large Workspaces:**
- Disable auto-analysis: Set `quantonium.autoAnalyze: false`
- Reduce animation frequency
- Limit concurrent windows

**Memory Usage:**
- Close unused tabs regularly
- Use session save/restore instead of keeping all windows open
- Monitor via System Monitor app

## 📚 API Reference

### QuantumAppController

**Key Methods:**
- `launch_app(app_name)` - Launch quantum application
- `close_tab(index)` - Close application tab
- `save_session()` - Save current session
- `load_session()` - Restore saved session

### WindowManager

**Key Methods:**
- `create_quantum_window(title, widget, size)` - Create new window
- `arrange_cascade()` - Cascade arrangement
- `arrange_tile_horizontal()` - Horizontal tile
- `arrange_tile_vertical()` - Vertical tile
- `save_session(path)` - Save session to file
- `load_session(path)` - Load session from file

### VS Code Extension API

**Available Commands:**
- `quantonium.launch` - Launch main application
- `quantonium.launch.{app}` - Launch specific app
- `quantonium.windows.{action}` - Window management
- `quantonium.analyze.quantum` - Analyze file
- `quantonium.session.{action}` - Session management

## 🚀 Advanced Usage

### Multi-Monitor Setup

**Configuration:**
```json
{
  "window_management": {
    "multi_monitor": true,
    "primary_monitor_apps": ["rft_visualizer", "quantum_crypto"],
    "secondary_monitor_apps": ["system_monitor", "q_notes"]
  }
}
```

**Usage:**
1. Connect multiple monitors
2. Configure app assignments in config
3. Windows automatically distribute across monitors

### Custom Themes

**Create New Theme:**
```css
/* In styles/my_theme.qss */
.quantum-glass {
    background: rgba(100, 50, 200, 0.1);
    border: 2px solid #6432c8;
    border-radius: 15px;
}
```

**Apply Theme:**
```python
# In quantum_app_controller.py
style_path = Path(__file__).parent.parent / "styles" / "my_theme.qss"
self.setStyleSheet(open(style_path).read())
```

### Batch Operations

**Launch Multiple Apps:**
```python
# Via Python API
controller = QuantumAppController()
apps = ['rft_visualizer', 'quantum_crypto', 'system_monitor']
for app in apps:
    controller.launch_app(app)
```

**VS Code Batch Commands:**
```javascript
// Custom batch launch
vscode.commands.executeCommand('quantonium.launch.rft');
vscode.commands.executeCommand('quantonium.launch.crypto');
vscode.commands.executeCommand('quantonium.windows.tile.horizontal');
```

## 🎯 Best Practices

### 1. Resource Management
- Close unused applications
- Use session management for complex setups
- Monitor system resources via built-in monitor

### 2. Workflow Integration
- Set up custom VS Code keybindings
- Use context menus for quick analysis
- Leverage auto-analysis for development

### 3. Performance
- Disable animations on slower systems
- Use targeted app launching vs. launching everything
- Regular session cleanup

### 4. Development
- Follow quantum styling conventions
- Use provided QSS classes
- Test with multiple monitor configurations

## 📞 Support

For issues, feature requests, or contributions:

1. **Documentation:** Check `QUANTONIUM_FRONTEND_DESIGN_MANUAL.md`
2. **Configuration:** Review `.quantonium/config.json`
3. **Logs:** Check `.quantonium/logs/` directory
4. **Debug:** Enable debug mode for detailed output

---

**🌌 Welcome to the Quantum Computing Interface Revolution! 🌌**
