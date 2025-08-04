# QuantoniumOS Test Vector Generator
# This script generates and publishes standardized test vectors for QuantoniumOS cryptographic primitives

Write-Host "QuantoniumOS Test Vector Generator" -ForegroundColor Cyan
Write-Host "=======================================" -ForegroundColor Cyan

# Navigate to the project root directory
$projectRoot = $PSScriptRoot | Split-Path -Parent
Set-Location $projectRoot

# Activate virtual environment if it exists
if (Test-Path "venv\Scripts\Activate.ps1") {
    Write-Host "Activating virtual environment..." -ForegroundColor Yellow
    & .\venv\Scripts\Activate.ps1
}

# Run the test vector generator
Write-Host "Running test vector generator..." -ForegroundColor Yellow
python scripts\generate_test_vectors.py

# Check if successful
if ($LASTEXITCODE -eq 0) {
    Write-Host "Test vectors successfully generated!" -ForegroundColor Green
} else {
    Write-Host "Error generating test vectors. Check the logs for details." -ForegroundColor Red
}

# Deactivate virtual environment if it was activated
if (Test-Path function:deactivate) {
    deactivate
}

Write-Host "Press any key to continue..."
$null = $host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")
