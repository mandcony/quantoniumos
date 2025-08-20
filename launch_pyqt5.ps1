# PowerShell Script to Launch QuantoniumOS PyQt5 Interface
# ============================================================

# Set encoding to UTF-8
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8

# Colors for output
$Color_Success = "Green"
$Color_Error = "Red"
$Color_Info = "Cyan"
$Color_Warning = "Yellow"

Write-Host "=" * 70 -ForegroundColor $Color_Info
Write-Host "    QuantoniumOS - PyQt5 Modern Desktop Interface" -ForegroundColor $Color_Info
Write-Host "=" * 70 -ForegroundColor $Color_Info

# Check if Python is available
try {
    $pythonVersion = python --version 2>$null
    if ($LASTEXITCODE -ne 0) {
        Write-Host "❌ Python not found in PATH" -ForegroundColor $Color_Error
        Write-Host "Please install Python 3.8+ and add it to your PATH" -ForegroundColor $Color_Warning
        Read-Host "Press Enter to exit"
        exit 1
    }
    Write-Host "✅ Python found: $pythonVersion" -ForegroundColor $Color_Success
} catch {
    Write-Host "❌ Error checking Python: $_" -ForegroundColor $Color_Error
    Read-Host "Press Enter to exit"
    exit 1
}

# Check if PyQt5 is installed
Write-Host "🔍 Checking PyQt5 installation..." -ForegroundColor $Color_Info
$pyqt5Check = python -c "import PyQt5; print('PyQt5 available')" 2>$null
if ($LASTEXITCODE -ne 0) {
    Write-Host "❌ PyQt5 not found" -ForegroundColor $Color_Error
    Write-Host "Installing PyQt5..." -ForegroundColor $Color_Info
    pip install PyQt5
    if ($LASTEXITCODE -ne 0) {
        Write-Host "❌ Failed to install PyQt5" -ForegroundColor $Color_Error
        Read-Host "Press Enter to exit"
        exit 1
    }
    Write-Host "✅ PyQt5 installed successfully" -ForegroundColor $Color_Success
} else {
    Write-Host "✅ PyQt5 is available" -ForegroundColor $Color_Success
}

# Launch the PyQt5 interface
Write-Host "🚀 Launching QuantoniumOS PyQt5 Interface..." -ForegroundColor $Color_Info
try {
    python launch_pyqt5.py
    if ($LASTEXITCODE -ne 0) {
        Write-Host "❌ QuantoniumOS PyQt5 interface exited with error code $LASTEXITCODE" -ForegroundColor $Color_Error
    } else {
        Write-Host "✅ QuantoniumOS PyQt5 interface closed successfully" -ForegroundColor $Color_Success
    }
} catch {
    Write-Host "❌ Error launching QuantoniumOS: $_" -ForegroundColor $Color_Error
}

Write-Host "👋 Thank you for using QuantoniumOS!" -ForegroundColor $Color_Info
Read-Host "Press Enter to exit"
