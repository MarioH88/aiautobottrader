import time
import os

def main():
    while True:
        with open("bot_flag.txt") as f:
            flag = f.read().strip()
        if flag == "start":
            # Run trading logic here
            print("Bot is running...")
            # ... trading code ...
        else:
            print("Bot is stopped.")
        time.sleep(10)

if __name__ == "__main__":
    main()