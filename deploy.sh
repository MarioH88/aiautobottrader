#!/bin/bash
# deploy.sh - Simple deployment script for your trading bot

# Install dependencies
pip3 install -r requirements.txt

# Start the bot in the background and log output
nohup python3 main.py > bot.log 2>&1 &
echo "Bot started. Logs in bot.log."
