import smtplib
from email.mime.text import MIMEText
import requests

def send_email(subject, body, to_email, from_email, smtp_server, smtp_port, smtp_user, smtp_pass):
    msg = MIMEText(body)
    msg['Subject'] = subject
    msg['From'] = from_email
    msg['To'] = to_email
    with smtplib.SMTP_SSL(smtp_server, smtp_port) as server:
        server.login(smtp_user, smtp_pass)
        server.sendmail(from_email, [to_email], msg.as_string())

def send_discord(message, webhook_url):
    data = {"content": message}
    requests.post(webhook_url, json=data)

if __name__ == "__main__":
    # Example usage (fill in your details)
    # send_email('Trade Alert', 'AAPL bought!', 'to@example.com', 'from@example.com', 'smtp.example.com', 465, 'user', 'pass')
    # send_discord('AAPL bought!', 'https://discord.com/api/webhooks/your_webhook_url')
    pass
