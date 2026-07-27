#!/bin/bash
# deploy.sh - Linux/macOS deployment for AI AutoBot Trader
set -euo pipefail
cd "$(dirname "$0")"

if [ ! -d ".venv" ]; then
  python3 -m venv .venv
fi
# shellcheck disable=SC1091
source .venv/bin/activate

pip install --upgrade pip
pip install -r requirements.txt

if [ ! -f .env ]; then
  cp .env.example .env
  echo "Created .env from .env.example — add Alpaca keys before trading."
fi

if [ ! -f model.pkl ]; then
  python train_model.py
fi

python -c "from bot.flag_file import write_flag; write_flag('bot_control.flag', 'start')"

nohup python main.py > bot.log 2>&1 &
echo "Bot started (PID $!). Logs: bot.log / bot_runtime.log"

if [ "${1:-}" = "--dashboard" ]; then
  nohup streamlit run dashboard_app.py --server.port=8501 --server.address=0.0.0.0 > dashboard.log 2>&1 &
  echo "Dashboard started on http://localhost:8501"
fi
