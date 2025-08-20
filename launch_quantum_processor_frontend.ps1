# PowerShell script to launch the 150 Qubit Quantum Processor Frontend

Write-Host "QuantoniumOS - Dynamic Quantum Processor Frontend Launcher" -ForegroundColor Green
Write-Host "=============================================================" -ForegroundColor Green

# Get the current directory
$currentDir = Get-Location

# Check if the quantum processor frontend exists
$frontendPath = Join-Path $currentDir "11_QUANTONIUMOS\apps\quantum_processor_frontend.py"
$launcherPath = Join-Path $currentDir "launch_quantum_processor_frontend.py"

if (Test-Path $frontendPath) {
    Write-Host "Found quantum processor frontend: $frontendPath" -ForegroundColor Yellow
} else {
    Write-Host "Quantum processor frontend not found at: $frontendPath" -ForegroundColor Red
    exit 1
}

if (Test-Path $launcherPath) {
    Write-Host "Found launcher: $launcherPath" -ForegroundColor Yellow
} else {
    Write-Host "Launcher not found at: $launcherPath" -ForegroundColor Red
    exit 1
}

# Try to find Python
$pythonCmd = $null
$pythonPaths = @("python", "python3", "py")

foreach ($cmd in $pythonPaths) {
    try {
        $version = & $cmd --version 2>$null
        if ($LASTEXITCODE -eq 0) {
            $pythonCmd = $cmd
            Write-Host "Found Python: $cmd ($version)" -ForegroundColor Green
            break
        }
    } catch {
        continue
    }
}

if (-not $pythonCmd) {
    Write-Host "Python not found. Please install Python or ensure it's in your PATH." -ForegroundColor Red
    exit 1
}

# Launch the frontend
Write-Host "Launching Dynamic Quantum Processor Frontend..." -ForegroundColor Cyan
Write-Host "Integrating with QuantoniumOS Quantum Kernel (up to 1000 qubits)" -ForegroundColor Yellow
Write-Host ""

try {
    & $pythonCmd $launcherPath
} catch {
    Write-Host "Error launching frontend: $_" -ForegroundColor Red
    exit 1
}

Write-Host ""
Write-Host "Frontend session ended." -ForegroundColor Yellow
