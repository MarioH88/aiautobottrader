# Morning routine: refresh training data, retrain model, smoke-test, start bot.
Set-Location $PSScriptRoot
$ErrorActionPreference = "Stop"
$venvPython = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $venvPython)) {
    $venvPython = "python"
}

Write-Host "=== 1/4 Build training data from Alpaca ===" -ForegroundColor Cyan
& $venvPython build_training_data.py
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host "=== 2/4 Train model ===" -ForegroundColor Cyan
& $venvPython train_model.py
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host "=== 3/4 Run tests (no live orders) ===" -ForegroundColor Cyan
& $venvPython test_bot.py
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host "=== 4/4 Start bot (Ctrl+C to stop) ===" -ForegroundColor Cyan
& $PSScriptRoot\start_bot.ps1
