# QuantoniumOS Memory Profiler
# This script profiles memory usage of QuantoniumOS cryptographic operations

Write-Host "QuantoniumOS Memory Profiler" -ForegroundColor Cyan
Write-Host "=======================================" -ForegroundColor Cyan

# Navigate to the project root directory
$projectRoot = $PSScriptRoot | Split-Path -Parent
Set-Location $projectRoot

# Ensure required packages are installed
Write-Host "Checking required packages..." -ForegroundColor Yellow
pip install psutil matplotlib pandas scipy

# Activate virtual environment if it exists
if (Test-Path "venv\Scripts\Activate.ps1") {
    Write-Host "Activating virtual environment..." -ForegroundColor Yellow
    & .\venv\Scripts\Activate.ps1
}

# Run the memory profiler
Write-Host "Running memory profiler..." -ForegroundColor Yellow
python core\testing\memory_profiler.py

# Check if successful
if ($LASTEXITCODE -eq 0) {
    Write-Host "Memory profiling completed successfully!" -ForegroundColor Green
} else {
    Write-Host "Error during memory profiling. Check the logs for details." -ForegroundColor Red
}

# Open the results directory
$resultsDir = Join-Path $projectRoot "memory_profiles"
if (Test-Path $resultsDir) {
    Write-Host "Opening results directory..." -ForegroundColor Yellow
    Start-Process $resultsDir
}

# Deactivate virtual environment if it was activated
if (Test-Path function:deactivate) {
    deactivate
}

Write-Host "Press any key to continue..."
$null = $host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")
