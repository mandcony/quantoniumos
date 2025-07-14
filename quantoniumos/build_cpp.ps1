# QuantoniumOS Robust C++ Build Script for Windows
# This script ensures consistent builds regardless of system Eigen installation

param(
    [switch]$Clean,
    [switch]$Debug,
    [string]$BuildDir = "build"
)

Write-Host "🔧 QuantoniumOS C++ Build Script" -ForegroundColor Cyan
Write-Host "=================================" -ForegroundColor Cyan

# Set error action preference
$ErrorActionPreference = "Stop"

# Clean previous build if requested
if ($Clean -and (Test-Path $BuildDir)) {
    Write-Host "🧹 Cleaning previous build..." -ForegroundColor Yellow
    Remove-Item -Recurse -Force $BuildDir
}

# Create build directory
if (-not (Test-Path $BuildDir)) {
    New-Item -ItemType Directory -Path $BuildDir | Out-Null
}

# Check for required tools
Write-Host "🔍 Checking build requirements..." -ForegroundColor Green

# Check for CMake
try {
    $cmakeVersion = cmake --version | Select-Object -First 1
    Write-Host "✅ Found CMake: $cmakeVersion" -ForegroundColor Green
} catch {
    Write-Host "❌ CMake not found. Please install CMake." -ForegroundColor Red
    exit 1
}

# Check for Visual Studio Build Tools or MSVC
$msvcFound = $false
try {
    $vsPath = &"${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe" -latest -property installationPath 2>$null
    if ($vsPath) {
        Write-Host "✅ Found Visual Studio at: $vsPath" -ForegroundColor Green
        $msvcFound = $true
    }
} catch {
    # Try alternative detection
    if (Get-Command "cl.exe" -ErrorAction SilentlyContinue) {
        Write-Host "✅ Found MSVC compiler (cl.exe)" -ForegroundColor Green
        $msvcFound = $true
    }
}

if (-not $msvcFound) {
    Write-Host "⚠️  MSVC not detected, trying with system default compiler..." -ForegroundColor Yellow
}

# Verify bundled Eigen
$eigenPath = "Eigen\eigen-3.4.0"
if (Test-Path $eigenPath) {
    Write-Host "✅ Found bundled Eigen at: $eigenPath" -ForegroundColor Green
} else {
    Write-Host "❌ Bundled Eigen not found at: $eigenPath" -ForegroundColor Red
    Write-Host "   Please ensure Eigen is properly extracted in the project root." -ForegroundColor Red
    exit 1
}

# Set build type
$buildType = if ($Debug) { "Debug" } else { "Release" }
Write-Host "🏗️  Build type: $buildType" -ForegroundColor Cyan

# Configure with CMake
Write-Host "⚙️  Configuring build..." -ForegroundColor Blue
Set-Location $BuildDir

try {
    if ($msvcFound) {
        # Use Visual Studio generator if available
        cmake -G "Visual Studio 17 2022" -A x64 -DCMAKE_BUILD_TYPE=$buildType .. 
    } else {
        # Fallback to MinGW or default generator
        cmake -DCMAKE_BUILD_TYPE=$buildType ..
    }
    
    if ($LASTEXITCODE -ne 0) {
        throw "CMake configuration failed"
    }
    
    Write-Host "✅ Configuration successful" -ForegroundColor Green
} catch {
    Write-Host "❌ Configuration failed: $_" -ForegroundColor Red
    Set-Location ..
    exit 1
}

# Build
Write-Host "🔨 Building C++ components..." -ForegroundColor Blue
try {
    cmake --build . --config $buildType --parallel
    
    if ($LASTEXITCODE -ne 0) {
        throw "Build failed"
    }
    
    Write-Host "✅ Build successful" -ForegroundColor Green
} catch {
    Write-Host "❌ Build failed: $_" -ForegroundColor Red
    Set-Location ..
    exit 1
}

# Test the executable
Write-Host "🧪 Testing executable..." -ForegroundColor Blue
$exePath = if ($buildType -eq "Debug") { "Debug\robust_test_symbolic.exe" } else { "Release\robust_test_symbolic.exe" }

if (-not (Test-Path $exePath)) {
    # Try alternative path
    $exePath = "robust_test_symbolic.exe"
}

if (Test-Path $exePath) {
    Write-Host "✅ Found executable: $exePath" -ForegroundColor Green
    
    try {
        Write-Host "🔍 Running tests..." -ForegroundColor Blue
        & ".\$exePath"
        
        if ($LASTEXITCODE -eq 0) {
            Write-Host "✅ All tests passed!" -ForegroundColor Green
        } else {
            Write-Host "⚠️  Some tests failed, but build is complete" -ForegroundColor Yellow
        }
    } catch {
        Write-Host "⚠️  Could not run tests: $_" -ForegroundColor Yellow
    }
} else {
    Write-Host "❌ Executable not found" -ForegroundColor Red
    Set-Location ..
    exit 1
}

Set-Location ..

Write-Host ""
Write-Host "🎉 Build process completed successfully!" -ForegroundColor Green
Write-Host "   Executable location: $BuildDir\$exePath" -ForegroundColor Cyan
Write-Host "   Use '-Clean' flag to clean build directory" -ForegroundColor Gray
Write-Host "   Use '-Debug' flag for debug build" -ForegroundColor Gray
