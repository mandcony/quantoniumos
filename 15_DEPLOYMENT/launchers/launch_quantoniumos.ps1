# QuantoniumOS PowerShell Launcher
# Complete system startup script for Windows PowerShell

param(
    [Parameter(Position=0)]
    [ValidateSet("desktop", "web", "headless", "test", "info")]
    [string]$Mode = "desktop",
    
    [int]$Port = 5000,
    [switch]$EnableDebug,
    [switch]$Help
)

function Show-Help {
    Write-Host @"
QuantoniumOS - Quantum Operating System Launcher

USAGE:
    .\launch_quantoniumos.ps1 [MODE] [OPTIONS]

MODES:
    desktop     Launch desktop interface (default)
    web         Launch web interface
    headless    Launch headless mode (API only)
    test        Run test suite
    info        Show system information

OPTIONS:
    -Port       Port for web mode (default: 5000)
    -EnableDebug Enable debug mode
    -Help       Show this help message

EXAMPLES:
    .\launch_quantoniumos.ps1
    .\launch_quantoniumos.ps1 web -Port 8080
    .\launch_quantoniumos.ps1 test
    .\launch_quantoniumos.ps1 info
"@
}

if ($Help) {
    Show-Help
    exit 0
}

Write-Host "================================================" -ForegroundColor Cyan
Write-Host "QuantoniumOS - Quantum Operating System" -ForegroundColor Cyan
Write-Host "================================================" -ForegroundColor Cyan
Write-Host ""

# Check Python installation
try {
    $pythonVersion = python --version 2>&1
    if ($LASTEXITCODE -ne 0) {
        throw "Python not found"
    }
    Write-Host "[OK] Python found: $pythonVersion" -ForegroundColor Green
} catch {
    Write-Host "[ERROR] Python is not installed or not in PATH" -ForegroundColor Red
    Write-Host "Please install Python 3.8+ and add it to your PATH" -ForegroundColor Yellow
    Read-Host "Press Enter to exit"
    exit 1
}

# Check if we're in the right directory
if (-not (Test-Path "quantoniumos.py")) {
    Write-Host "[ERROR] quantoniumos.py not found" -ForegroundColor Red
    Write-Host "Please run this script from the QuantoniumOS root directory" -ForegroundColor Yellow
    Read-Host "Press Enter to exit"
    exit 1
}

# Set environment variables
$pathElements = @(
    $PWD,
    (Join-Path $PWD "kernel"),
    (Join-Path $PWD "gui"),
    (Join-Path $PWD "web"),
    (Join-Path $PWD "filesystem"),
    (Join-Path $PWD "apps"),
    (Join-Path $PWD "phase3"),
    (Join-Path $PWD "phase4"),
    (Join-Path $PWD "11_QUANTONIUMOS")
)
$env:PYTHONPATH = $pathElements -join ";"
$env:QUANTONIUMOS_ROOT = $PWD

Write-Host "Environment configured" -ForegroundColor Green
Write-Host "PYTHONPATH: $env:PYTHONPATH" -ForegroundColor Gray
Write-Host ""

# Create requirements.txt if it doesn't exist
if (-not (Test-Path "requirements.txt")) {
    Write-Host "Creating basic requirements.txt..." -ForegroundColor Yellow
    @"
numpy>=1.21.0
flask>=2.0.0
cryptography>=3.4.0
matplotlib>=3.5.0
scipy>=1.7.0
"@ | Out-File -FilePath "requirements.txt" -Encoding UTF8
}

# Check dependencies
Write-Host "Checking dependencies..." -ForegroundColor Yellow
try {
    python -c "import numpy, flask, cryptography" 2>$null
    if ($LASTEXITCODE -ne 0) {
        throw "Missing dependencies"
    }
    Write-Host "[OK] All dependencies available" -ForegroundColor Green
} catch {
    Write-Host "Installing required packages..." -ForegroundColor Yellow
    pip install -r requirements.txt
    if ($LASTEXITCODE -ne 0) {
        Write-Host "[ERROR] Failed to install dependencies" -ForegroundColor Red
        Read-Host "Press Enter to exit"
        exit 1
    }
    Write-Host "[OK] Dependencies installed successfully" -ForegroundColor Green
}

Write-Host ""
Write-Host "Starting QuantoniumOS in $Mode mode..." -ForegroundColor Cyan
Write-Host ""

# Build command arguments
$cmdArgs = @("quantoniumos.py", $Mode)

if ($Mode -eq "web") {
    $cmdArgs += "--port", $Port
    if ($EnableDebug) {
        $cmdArgs += "--debug"
    }
}

# Launch QuantoniumOS
try {
    switch ($Mode) {
        "desktop" {
            Write-Host "[DESKTOP] Launching Desktop Interface..." -ForegroundColor Blue
        }
        "web" {
            Write-Host "[WEB] Launching Web Interface on port $Port..." -ForegroundColor Blue
            if ($EnableDebug) {
                Write-Host "   Debug mode enabled" -ForegroundColor Yellow
            }
        }
        "headless" {
            Write-Host "[HEADLESS] Launching Headless Mode..." -ForegroundColor Blue
        }
        "test" {
            Write-Host "[TEST] Running Test Suite..." -ForegroundColor Blue
        }
        "info" {
            Write-Host "[INFO] Displaying System Information..." -ForegroundColor Blue
        }
    }
    
    python @cmdArgs
    
    if ($LASTEXITCODE -ne 0) {
        throw "QuantoniumOS exited with error code $LASTEXITCODE"
    }
    
    Write-Host ""
    Write-Host "[OK] QuantoniumOS shutdown complete" -ForegroundColor Green
    
} catch {
    Write-Host ""
    Write-Host "[ERROR] QuantoniumOS failed to start" -ForegroundColor Red
    Write-Host "Error: $_" -ForegroundColor Red
    Write-Host "Check the logs above for details" -ForegroundColor Yellow
    Read-Host "Press Enter to exit"
    exit 1
}

if ($Mode -ne "headless") {
    Read-Host "Press Enter to exit"
}
