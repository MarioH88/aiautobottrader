# Start the auto-trading bot (runs until you close this window or press Ctrl+C)
Set-Location $PSScriptRoot
$venvPython = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $venvPython)) {
    $venvPython = "python"
}
& $venvPython -c "from bot.flag_file import write_flag; write_flag('bot_control.flag', 'start')"
& $venvPython main.py
