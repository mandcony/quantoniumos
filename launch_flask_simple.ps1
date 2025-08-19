# QuantoniumOS Flask App Launcher for PowerShell

param(
    [string]$Mode = "normal",
    [string]$ServerHost = "127.0.0.1",
    [int]$Port = 5000
)

Write-Host "QuantoniumOS Flask App Launcher" -ForegroundColor Cyan
Write-Host "================================" -ForegroundColor Cyan

$RootPath = Split-Path -Parent $PSScriptRoot
$RFTPath = Join-Path $RootPath "04_RFT_ALGORITHMS"
$AppPath = Join-Path $RootPath "03_RUNNING_SYSTEMS"

# Python command to run Flask app
$PythonScript = @"
import sys
sys.path.extend(['$($RFTPath -replace '\\', '/')','$($AppPath -replace '\\', '/')'])
import os
os.chdir('$($AppPath -replace '\\', '/')')
from main import create_app
app = create_app()
if __name__ == '__main__':
    print('🚀 Starting QuantoniumOS Flask Application...')
    print('📍 Host: $ServerHost')
    print('📍 Port: $Port')
    print('📍 Access: http://$ServerHost`:$Port')
    app.run(host='$ServerHost', port=$Port, debug=True)
"@

switch ($Mode.ToLower()) {
    "background" {
        Write-Host "Starting Flask app in background..." -ForegroundColor Green
        Start-Process python -ArgumentList "-c", $PythonScript -NoNewWindow
        Write-Host "Flask app started in background" -ForegroundColor Green
        Write-Host "Access: http://$ServerHost`:$Port" -ForegroundColor Cyan
        break
    }
    
    "job" {
        Write-Host "Starting Flask app as PowerShell job..." -ForegroundColor Green
        $Job = Start-Job -ScriptBlock {
            param($Script)
            python -c $Script
        } -ArgumentList $PythonScript
        
        Write-Host "Flask app started as job (ID: $($Job.Id))" -ForegroundColor Green
        Write-Host "Access: http://$ServerHost`:$Port" -ForegroundColor Cyan
        Write-Host "Use 'Get-Job' and 'Receive-Job $($Job.Id)' to monitor" -ForegroundColor Yellow
        break
    }
    
    "newwindow" {
        Write-Host "Starting Flask app in new window..." -ForegroundColor Green
        Start-Process python -ArgumentList "-c", $PythonScript
        Write-Host "Flask app started in new window" -ForegroundColor Green
        Write-Host "Access: http://$ServerHost`:$Port" -ForegroundColor Cyan
        break
    }
    
    "help" {
        Write-Host "Usage Examples:" -ForegroundColor Yellow
        Write-Host "Normal:      .\launch_flask.ps1" -ForegroundColor White
        Write-Host "Background:  .\launch_flask.ps1 -Mode background" -ForegroundColor White
        Write-Host "Job:         .\launch_flask.ps1 -Mode job" -ForegroundColor White
        Write-Host "New window:  .\launch_flask.ps1 -Mode newwindow" -ForegroundColor White
        Write-Host "Custom:      .\launch_flask.ps1 -ServerHost 0.0.0.0 -Port 8080" -ForegroundColor White
        Write-Host ""
        Write-Host "PowerShell Background Commands:" -ForegroundColor Yellow
        Write-Host "Wrong:   python script.py &" -ForegroundColor Red
        Write-Host "Correct: Start-Process python -ArgumentList 'script.py' -NoNewWindow" -ForegroundColor Green
        break
    }
    
    default {
        Write-Host "Starting Flask app normally (foreground)..." -ForegroundColor Green
        Write-Host "Use Ctrl+C to stop the server" -ForegroundColor Yellow
        python -c $PythonScript
        break
    }
}
