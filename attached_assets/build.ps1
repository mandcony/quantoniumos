$ErrorActionPreference = "Stop"

# Configuration
$BuildDir = "C:\quantonium_os\bin"
$EigenInclude = "C:\quantonium_os\Eigen\eigen-3.4.0"
$Gxx = "g++"
$CommonFlags = "-O2 -fPIC -I $EigenInclude"
$LinkFlags = "-shared -fPIC -lstdc++ -lgcc -shared-libgcc"

# Ensure output directory exists
if (-not (Test-Path $BuildDir)) {
    New-Item -Path $BuildDir -ItemType Directory | Out-Null
}

# Function to build a DLL
function Build-DLL {
    param (
        [string]$SourceFile,
        [string]$DllName
    )
    $ObjectFile = [IO.Path]::ChangeExtension($SourceFile, ".o")
    $DllPath = Join-Path $BuildDir $DllName

    Write-Host "Building $DllName..." -ForegroundColor Cyan

    # Compile source to object
    $compileCmd = "$Gxx -c $SourceFile -o $ObjectFile $CommonFlags -DBUILDING_DLL"
    Write-Host "Executing: $compileCmd" -ForegroundColor Yellow
    Invoke-Expression $compileCmd
    if ($LASTEXITCODE -ne 0) {
        Write-Host "Compilation failed for $SourceFile!" -ForegroundColor Red
        exit 1
    }

    # Link object to DLL
    $linkCmd = "$Gxx -shared -o $DllPath $ObjectFile $LinkFlags"
    Write-Host "Executing: $linkCmd" -ForegroundColor Yellow
    Invoke-Expression $linkCmd
    if ($LASTEXITCODE -ne 0) {
        Write-Host "Linking failed for $DllName!" -ForegroundColor Red
        exit 1
    }

    Write-Host "Build successful for $DllName!" -ForegroundColor Green
}

# Build both DLLs
try {
    Build-DLL -SourceFile "quantum_os.cpp" -DllName "quantum_os.dll"
    Build-DLL -SourceFile "symbolic_eigenvector.cpp" -DllName "engine_core.dll"
} catch {
    Write-Host "Unexpected error: $_" -ForegroundColor Red
    exit 1
}

# Copy dependencies
$MinGWBin = "C:\mingw64\bin"
$Dependencies = @("libwinpthread-1.dll", "libstdc++-6.dll", "libgcc_s_seh-1.dll")
foreach ($dep in $Dependencies) {
    $src = Join-Path $MinGWBin $dep
    if (Test-Path $src) {
        Copy-Item $src -Destination $BuildDir -Force
        Write-Host "Copied dependency: $dep" -ForegroundColor Yellow
    }
}