# QuantoniumOS Flask App Launcher for PowerShell
# 
# This script demonstrates proper PowerShell syntax for running background processes
# and launching the QuantoniumOS Flask application.

param(
    [string]$Mode = "normal",  # normal, background, job, or newwindow
    [string]$ServerHost = "127.0.0.1",
    [int]$Port = 5000
)

Write-Host "QuantoniumOS Flask App Launcher" -ForegroundColor Cyan
Write-Host "================================" -ForegroundColor Cyan

# Set up paths
$RootPath = Split-Path -Parent $PSScriptRoot
$RFTPath = Join-Path $RootPath "04_RFT_ALGORITHMS"
$AppPath = Join-Path $RootPath "03_RUNNING_SYSTEMS"

Write-Host "Root Path: $RootPath" -ForegroundColor Yellow
Write-Host "RFT Path: $RFTPath" -ForegroundColor Yellow
Write-Host "App Path: $AppPath" -ForegroundColor Yellow

# Verify paths exist
if (-not (Test-Path $RFTPath)) {
    Write-Host "❌ ERROR: RFT algorithms path not found: $RFTPath" -ForegroundColor Red
    exit 1
}

if (-not (Test-Path $AppPath)) {
    Write-Host "❌ ERROR: App path not found: $AppPath" -ForegroundColor Red
    exit 1
}

# Python command to run the Flask app
$PythonCommand = @"
import sys
sys.path.extend(['$($RFTPath -replace '\\', '\\')','$($AppPath -replace '\\', '\\')'])
import os
os.chdir('$($AppPath -replace '\\', '\\')')
from main import create_app
app = create_app()
if __name__ == '__main__':
    print('🚀 Starting QuantoniumOS Flask Application...')
    print(f'📍 Host: $ServerHost')
    print(f'📍 Port: $Port')
    print('📍 Access: http://$ServerHost`:$Port')
    app.run(host='$ServerHost', port=$Port, debug=True)
"@

switch ($Mode.ToLower()) {
    "background" {
        Write-Host "🔄 Starting Flask app in background (Start-Process)..." -ForegroundColor Green
        Start-Process python -ArgumentList "-c", "`"$PythonCommand`"" -NoNewWindow
        Write-Host "✅ Flask app started in background process" -ForegroundColor Green
        Write-Host "📍 Access: http://$ServerHost`:$Port" -ForegroundColor Cyan
    }
    
    "job" {
        Write-Host "🔄 Starting Flask app as PowerShell job..." -ForegroundColor Green
        $Job = Start-Job -ScriptBlock {
            param($Command)
            python -c $Command
        } -ArgumentList $PythonCommand
        
        Write-Host "✅ Flask app started as job (ID: $($Job.Id))" -ForegroundColor Green
        Write-Host "📍 Access: http://$ServerHost`:$Port" -ForegroundColor Cyan
        Write-Host "💡 Use 'Get-Job' and 'Receive-Job $($Job.Id)' to monitor" -ForegroundColor Yellow
    }
    
    "newwindow" {
        Write-Host "🔄 Starting Flask app in new window..." -ForegroundColor Green
        Start-Process python -ArgumentList "-c", "`"$PythonCommand`""
        Write-Host "✅ Flask app started in new window" -ForegroundColor Green
        Write-Host "📍 Access: http://$ServerHost`:$Port" -ForegroundColor Cyan
    }
    
    default {
        Write-Host "🔄 Starting Flask app normally (foreground)..." -ForegroundColor Green
        Write-Host "💡 Use Ctrl+C to stop the server" -ForegroundColor Yellow
        python -c $PythonCommand
    }
}

# Show usage examples
if ($Mode -eq "help") {
    Write-Host "`nUsage Examples:" -ForegroundColor Cyan
    Write-Host "=================" -ForegroundColor Cyan
    Write-Host "Normal (foreground):    .\launch_flask.ps1" -ForegroundColor White
    Write-Host "Background process:     .\launch_flask.ps1 -Mode background" -ForegroundColor White
    Write-Host "PowerShell job:         .\launch_flask.ps1 -Mode job" -ForegroundColor White
    Write-Host "New window:             .\launch_flask.ps1 -Mode newwindow" -ForegroundColor White
    Write-Host "Custom host/port:       .\launch_flask.ps1 -ServerHost 0.0.0.0 -Port 8080" -ForegroundColor White
    Write-Host "`nPowerShell Background Process Commands:" -ForegroundColor Yellow
    Write-Host "❌ Wrong (Linux/Bash):   python script.py &" -ForegroundColor Red
    Write-Host "✅ Correct (PowerShell): Start-Process python -ArgumentList 'script.py' -NoNewWindow" -ForegroundColor Green
    Write-Host "✅ Alternative:          Start-Job -ScriptBlock { python script.py }" -ForegroundColor Green
}
