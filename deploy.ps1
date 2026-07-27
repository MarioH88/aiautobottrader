# Deploy / start AI AutoBot Trader on Windows
# Usage:
#   .\deploy.ps1              # install deps + start bot
#   .\deploy.ps1 -Dashboard   # also start Streamlit dashboard
#   .\deploy.ps1 -Docker      # run via docker compose

param(
    [switch]$Dashboard,
    [switch]$Docker,
    [switch]$SkipInstall
)

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

if ($Docker) {
    if (-not (Get-Command docker -ErrorAction SilentlyContinue)) {
        Write-Error "Docker is not installed or not on PATH."
    }
    docker compose up -d --build
    Write-Host "Bot + dashboard started via Docker Compose."
    Write-Host "Dashboard: http://localhost:8501"
    exit 0
}

$venvPython = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $venvPython)) {
    Write-Host "Creating virtual environment..."
    $systemPython = "$env:LOCALAPPDATA\Programs\Python\Python312\python.exe"
    if (-not (Test-Path $systemPython)) {
        $systemPython = "python"
    }
    & $systemPython -m venv .venv
    $venvPython = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
}

if (-not $SkipInstall) {
    Write-Host "Installing dependencies..."
    & $venvPython -m pip install --upgrade pip
    & $venvPython -m pip install -r requirements.txt
}

if (-not (Test-Path ".env")) {
    Copy-Item ".env.example" ".env"
    Write-Warning "Created .env from .env.example — add your Alpaca keys before trading."
}

if (-not (Test-Path "model.pkl")) {
    Write-Host "Training ML model..."
    & $venvPython train_model.py
}

& $venvPython -c "from bot.flag_file import write_flag; write_flag('bot_control.flag', 'start')"

Write-Host "Starting trading bot..."
Start-Process -FilePath $venvPython -ArgumentList "main.py" -WorkingDirectory $PSScriptRoot -WindowStyle Minimized

if ($Dashboard) {
    Write-Host "Starting Streamlit dashboard on http://localhost:8501 ..."
    Start-Process -FilePath $venvPython -ArgumentList @("-m", "streamlit", "run", "dashboard_app.py", "--server.port=8501") -WorkingDirectory $PSScriptRoot
}

Write-Host "Deploy complete."
Write-Host "  Bot control: Set-Content bot_control.flag -Value stop -Encoding utf8 -NoNewline"
Write-Host "  Logs: bot_runtime.log / trade_log.csv"
