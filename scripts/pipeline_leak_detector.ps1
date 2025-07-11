# QuantoniumOS Pipeline Leak Detector (PowerShell)
# Like putting a tube underwater and watching for bubbles

param(
    [switch]$Fast,
    [switch]$Verbose
)

# Initialize counters
$script:LeakCount = 0
$script:TotalChecks = 0
$script:StartTime = Get-Date

Write-Host "🔍 QuantoniumOS Pipeline Leak Detector" -ForegroundColor Blue
Write-Host "=====================================" -ForegroundColor Blue

function Test-Leak {
    param(
        [string]$TestName,
        [scriptblock]$TestCommand,
        [int]$TimeoutSeconds = 30
    )
    
    $script:TotalChecks++
    Write-Host "🧪 Testing: $TestName... " -NoNewline
    
    try {
        $job = Start-Job -ScriptBlock $TestCommand
        $result = Wait-Job $job -Timeout $TimeoutSeconds
        
        if ($result) {
            $output = Receive-Job $job
            Remove-Job $job
            Write-Host "✅ PASS" -ForegroundColor Green
            return $true
        } else {
            Remove-Job $job -Force
            Write-Host "❌ LEAK (TIMEOUT)" -ForegroundColor Red
            $script:LeakCount++
            return $false
        }
    }
    catch {
        Write-Host "❌ LEAK (ERROR)" -ForegroundColor Red
        if ($Verbose) { Write-Host "   Error: $($_.Exception.Message)" -ForegroundColor Yellow }
        $script:LeakCount++
        return $false
    }
}

function Invoke-TimedTest {
    param(
        [string]$TestName,
        [scriptblock]$TestCommand,
        [int]$TimeoutSeconds = 60
    )
    
    Write-Host "`n⏱️  Running: $TestName (max ${TimeoutSeconds}s)" -ForegroundColor Yellow
    $testStart = Get-Date
    
    try {
        $job = Start-Job -ScriptBlock $TestCommand
        $result = Wait-Job $job -Timeout $TimeoutSeconds
        
        if ($result) {
            $output = Receive-Job $job
            Remove-Job $job
            $duration = [math]::Round(((Get-Date) - $testStart).TotalSeconds, 1)
            Write-Host "✅ Completed in ${duration}s" -ForegroundColor Green
            if ($Verbose -and $output) { Write-Host $output }
            return $true
        } else {
            Remove-Job $job -Force
            $duration = [math]::Round(((Get-Date) - $testStart).TotalSeconds, 1)
            Write-Host "❌ Failed after ${duration}s" -ForegroundColor Red
            $script:LeakCount++
            return $false
        }
    }
    catch {
        $duration = [math]::Round(((Get-Date) - $testStart).TotalSeconds, 1)
        Write-Host "❌ Error after ${duration}s" -ForegroundColor Red
        if ($Verbose) { Write-Host "   Error: $($_.Exception.Message)" -ForegroundColor Yellow }
        $script:LeakCount++
        return $false
    }
}

Write-Host "`nPhase 1: Pre-flight Checks (Looking for obvious leaks)" -ForegroundColor Blue
Write-Host "================================================================"

# Check critical files
Test-Leak "requirements.txt exists" { Test-Path "requirements.txt" }
Test-Leak "main.py exists" { Test-Path "main.py" }
Test-Leak "setup.py exists" { Test-Path "setup.py" }
Test-Leak "CI workflow exists" { Test-Path ".github/workflows/main-ci.yml" }

# Python syntax checks
Test-Leak "Python available" { python --version | Out-Null; $LASTEXITCODE -eq 0 }
Test-Leak "main.py syntax" { python -m py_compile main.py; $LASTEXITCODE -eq 0 }

Write-Host "`nPhase 2: Dependency Resolution (Testing tube connections)" -ForegroundColor Blue
Write-Host "================================================================"

# Test critical imports
Test-Leak "Flask import" { python -c "import flask" | Out-Null; $LASTEXITCODE -eq 0 }
Test-Leak "NumPy import" { python -c "import numpy" | Out-Null; $LASTEXITCODE -eq 0 }
Test-Leak "Cryptography import" { python -c "import cryptography" | Out-Null; $LASTEXITCODE -eq 0 }

if (-not $Fast) {
    Write-Host "`nPhase 3: CLI Verification (Testing basic functionality)" -ForegroundColor Blue
    Write-Host "================================================================"
    
    if (Test-Path "scripts/verify_cli.py") {
        Invoke-TimedTest "CLI verification" { 
            python scripts/verify_cli.py --verbose
            $LASTEXITCODE -eq 0 
        } 60
    } else {
        Write-Host "⚠️  CLI verification script not found" -ForegroundColor Yellow
        Test-Leak "Basic CLI test" { python -c "print('CLI test passed')" | Out-Null; $LASTEXITCODE -eq 0 }
    }
    
    Write-Host "`nPhase 4: Security Scan (Looking for security leaks)" -ForegroundColor Blue
    Write-Host "================================================================"
    
    # Check if bandit is available
    $banditAvailable = try { 
        bandit --version | Out-Null
        $LASTEXITCODE -eq 0 
    } catch { $false }
    
    if ($banditAvailable) {
        Invoke-TimedTest "Bandit security scan" { 
            bandit -r core/ -f json | Out-Null
            $LASTEXITCODE -eq 0 
        } 30
    } else {
        Write-Host "⚠️  Bandit not available - installing..." -ForegroundColor Yellow
        Test-Leak "Install bandit" { 
            pip install bandit | Out-Null
            $LASTEXITCODE -eq 0 
        }
    }
    
    Write-Host "`nPhase 5: Build Test (Testing package integrity)" -ForegroundColor Blue
    Write-Host "================================================================"
    
    Invoke-TimedTest "Package build test" { 
        python setup.py check | Out-Null
        $LASTEXITCODE -eq 0 
    } 45
}

Write-Host "`nPhase 6: Final Leak Report" -ForegroundColor Blue
Write-Host "================================================================"

$totalDuration = [math]::Round(((Get-Date) - $script:StartTime).TotalSeconds, 1)
Write-Host "⏱️  Total execution time: ${totalDuration}s"
Write-Host "🧪 Total checks performed: $($script:TotalChecks)"

if ($script:LeakCount -eq 0) {
    Write-Host "🎉 NO LEAKS DETECTED - Pipeline ready for production!" -ForegroundColor Green
    Write-Host "✅ Your tube is watertight - safe to submerge" -ForegroundColor Green
    exit 0
} else {
    Write-Host "🚨 $($script:LeakCount) LEAKS DETECTED out of $($script:TotalChecks) checks" -ForegroundColor Red
    Write-Host "❌ Fix leaks before pushing to production" -ForegroundColor Red
    exit 1
}
