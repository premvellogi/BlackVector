<#
.SYNOPSIS
    SatQuery AI - Full Stack Launch Script
    Starts all 4 services in the correct order for local development.

.DESCRIPTION
    Services started:
      1. B2 Validation Service  (port 8100) - input validation & confidence
      2. ML Service             (port 8200) - InternVL2 real GPU inference
      3. Controller B1          (port 8000) - orchestrator / state machine
      4. Frontend               (port 5173) - React + Vite dev server

.USAGE
    .\run_satquery.ps1
    .\run_satquery.ps1 -MockML    # Use mock ML instead of real GPU
#>

param(
    [switch]$MockML  # Use mock ML service instead of real GPU inference
)

$ErrorActionPreference = "Continue"

# --- Paths ----------------------------------------------------
$NetraRoot       = $PSScriptRoot
$ControllerRoot  = $PSScriptRoot
$FrontendRoot    = "$NetraRoot\satquery_ui"
$PythonExe       = "python" # Uses active virtual environment or python executable
if (Test-Path "$NetraRoot\satquery_env\Scripts\python.exe") {
    $PythonExe   = "$NetraRoot\satquery_env\Scripts\python.exe"
}

Write-Host ""
Write-Host "=== SatQuery AI - Full Stack Launcher ===" -ForegroundColor Cyan
Write-Host "  B2 Validation   -> http://localhost:8100" -ForegroundColor DarkGray
Write-Host "  ML Service      -> http://localhost:8200" -ForegroundColor DarkGray
Write-Host "  Controller B1   -> http://localhost:8000" -ForegroundColor DarkGray
Write-Host "  Frontend        -> http://localhost:5173" -ForegroundColor DarkGray
Write-Host ""

# --- 1. Start B2 Validation Service (port 8100) ---------------
Write-Host "[1/4] Starting B2 Validation Service on :8100..." -ForegroundColor Green
Start-Process powershell -ArgumentList @(
    "-NoExit", "-Command",
    "Set-Location '$ControllerRoot'; & '$PythonExe' -m uvicorn mock_b2:app --host 0.0.0.0 --port 8100"
) -WindowStyle Normal

Start-Sleep -Seconds 2

# --- 2. Start ML Service (port 8200) -------------------------
if ($MockML) {
    Write-Host "[2/4] Starting MOCK ML Service on :8200..." -ForegroundColor Yellow
    Start-Process powershell -ArgumentList @(
        "-NoExit", "-Command",
        "Set-Location '$ControllerRoot'; & '$PythonExe' -m uvicorn mock_ml.app:app --host 0.0.0.0 --port 8200"
    ) -WindowStyle Normal
} else {
    Write-Host "[2/4] Starting REAL ML Service (EOV2B GPU) on :8200..." -ForegroundColor Magenta
    Start-Process powershell -ArgumentList @(
        "-NoExit", "-Command",
        "Set-Location '$NetraRoot'; & '$PythonExe' -m uvicorn services.models.serving.server:app --host 0.0.0.0 --port 8200"
    ) -WindowStyle Normal
}

Start-Sleep -Seconds 3

# --- 3. Start Controller B1 (port 8000) ----------------------
Write-Host "[3/4] Starting Controller B1 on :8000..." -ForegroundColor Green

Start-Process powershell -ArgumentList @(
    "-NoExit", "-Command",
    "`$env:SATQUERY_B2_BASE_URL='http://localhost:8100'; `$env:SATQUERY_ML_BASE_URL='http://localhost:8200'; `$env:SATQUERY_CONFIDENCE_BASE_URL='http://localhost:8100'; Set-Location '$ControllerRoot'; & '$PythonExe' -m uvicorn controller.main:app --host 0.0.0.0 --port 8000 --reload"
) -WindowStyle Normal

Start-Sleep -Seconds 2

# --- 4. Start Frontend (port 5173) ---------------------------
Write-Host "[4/4] Starting Frontend on :5173..." -ForegroundColor Green
Start-Process powershell -ArgumentList @(
    "-NoExit", "-Command",
    "Set-Location '$FrontendRoot'; npm run dev -- --host --port 5173"
) -WindowStyle Normal

Start-Sleep -Seconds 3

# --- Status ---------------------------------------------------
Write-Host ""
Write-Host "All services launched! Open your browser at:" -ForegroundColor White
Write-Host "   http://localhost:5173" -ForegroundColor Yellow
Write-Host ""
Write-Host "Backend health checks:" -ForegroundColor DarkGray
Write-Host "   B2:         http://localhost:8100/health" -ForegroundColor DarkGray
Write-Host "   ML:         http://localhost:8200/health" -ForegroundColor DarkGray
Write-Host "   Controller: http://localhost:8000/health" -ForegroundColor DarkGray
Write-Host ""
