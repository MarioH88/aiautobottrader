# Start the auto-trading bot (runs until you close this window or press Ctrl+C)
Set-Location $PSScriptRoot
python -c "from bot.flag_file import write_flag; write_flag('bot_control.flag', 'start')"
python main.py
