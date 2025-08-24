# QuantoniumOS - Unified Launcher
# ================================
# Single PowerShell script to launch the complete QuantoniumOS

# Set console encoding
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8

# Colors
$Green = "Green"
$Red = "Red"
$Cyan = "Cyan"
$Yellow = "Yellow"

Write-Host "=" * 70 -ForegroundColor $Cyan
Write-Host "    🚀 QuantoniumOS - Complete Quantum Operating System" -ForegroundColor $Cyan
Write-Host "=" * 70 -ForegroundColor $Cyan

# Check Python
Write-Host "🔍 Checking Python installation..." -ForegroundColor $Cyan
try {
    $pythonVersion = python --version 2>$null
    if ($LASTEXITCODE -eq 0) {
        Write-Host "✅ $pythonVersion" -ForegroundColor $Green
    } else {
        Write-Host "❌ Python not found. Please install Python 3.8+" -ForegroundColor $Red
        Read-Host "Press Enter to exit"
        exit 1
    }
} catch {
    Write-Host "❌ Error checking Python: $_" -ForegroundColor $Red
    Read-Host "Press Enter to exit"
    exit 1
}

# Launch QuantoniumOS
Write-Host "🚀 Starting QuantoniumOS..." -ForegroundColor $Cyan
Write-Host "⚛️ Loading quantum kernel..." -ForegroundColor $Yellow
Write-Host "📜 Initializing patent modules..." -ForegroundColor $Yellow
Write-Host "🖥️ Opening desktop interface..." -ForegroundColor $Yellow

try {
    python start_quantoniumos.py
    if ($LASTEXITCODE -eq 0) {
        Write-Host "✅ QuantoniumOS closed successfully" -ForegroundColor $Green
    } else {
        Write-Host "❌ QuantoniumOS exited with error" -ForegroundColor $Red
    }
} catch {
    Write-Host "❌ Error launching QuantoniumOS: $_" -ForegroundColor $Red
}

Write-Host "👋 Thank you for using QuantoniumOS!" -ForegroundColor $Cyan
Read-Host "Press Enter to exit"
