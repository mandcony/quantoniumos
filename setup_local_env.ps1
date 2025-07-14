# setup_local_env.ps1
# QuantoniumOS Local Configuration Script

Write-Host "Setting up QuantoniumOS local environment..." -ForegroundColor Cyan

# Create a directory for local data if it doesn't exist
$dataDir = Join-Path $PSScriptRoot "quantoniumos\instance"
if (-not (Test-Path $dataDir)) {
    New-Item -ItemType Directory -Path $dataDir -Force | Out-Null
    Write-Host "Created local data directory: $dataDir" -ForegroundColor Green
}

# Generate secure random keys
function Generate-SecureKey {
    $random = [System.Security.Cryptography.RandomNumberGenerator]::Create()
    $bytes = New-Object byte[] 32
    $random.GetBytes($bytes)
    return [Convert]::ToBase64String($bytes)
}

# Set environment variables
$masterKey = Generate-SecureKey
$dbEncryptionKey = Generate-SecureKey
$jwtSecret = Generate-SecureKey

# Set environment variables for the current PowerShell session
$env:QUANTONIUM_MASTER_KEY = $masterKey
$env:DATABASE_ENCRYPTION_KEY = $dbEncryptionKey
$env:JWT_SECRET_KEY = $jwtSecret
$env:FLASK_ENV = "development"
$env:SQLALCHEMY_DATABASE_URI = "sqlite:///$dataDir/quantonium.db"
$env:REDIS_DISABLED = "true"
$env:PORT = "5000"

# Create a .env file for future use
$envFile = Join-Path $PSScriptRoot "quantoniumos\.env"
@"
# QuantoniumOS Local Environment Configuration
# Generated on $(Get-Date)

# Encryption Keys
QUANTONIUM_MASTER_KEY=$masterKey
DATABASE_ENCRYPTION_KEY=$dbEncryptionKey
JWT_SECRET_KEY=$jwtSecret

# Database Configuration
SQLALCHEMY_DATABASE_URI=sqlite:///instance/quantonium.db

# Service Configuration
FLASK_ENV=development
REDIS_DISABLED=true
PORT=5000

# Fix for routing conflicts
QUANTONIUM_ROUTE_FIX=true
"@ | Out-File -FilePath $envFile -Encoding utf8

Write-Host "Created .env file: $envFile" -ForegroundColor Green
Write-Host "Environment variables set for current session" -ForegroundColor Green

# Output instructions
Write-Host "`nTo run QuantoniumOS with these settings:" -ForegroundColor Yellow
Write-Host "1. Run this script first: .\setup_local_env.ps1" -ForegroundColor Yellow
Write-Host "2. Then start the application: cd quantoniumos; python app.py" -ForegroundColor Yellow
Write-Host "`nEnvironment ready for QuantoniumOS!" -ForegroundColor Cyan
