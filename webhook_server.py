from flask import Flask, request

app = Flask(__name__)

@app.route('/tradingview-webhook', methods=['POST'])
def tradingview_webhook():
    data = request.json
    print("Received TradingView alert:", data)
    # Add your trading logic here
    return 'Webhook received', 200

if __name__ == '__main__':
    app.run(port=5000)ms-appid:undefined