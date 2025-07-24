"""
Sample: Reddit Sentiment Fetcher for Trading Bot
- Uses PRAW to fetch posts/comments for ticker mentions
- Analyzes sentiment using TextBlob
- Requires Reddit API credentials
"""
import praw
from textblob import TextBlob

def get_reddit_sentiment(ticker, client_id, client_secret, user_agent):
    reddit = praw.Reddit(
        client_id=client_id,
        client_secret=client_secret,
        user_agent=user_agent
    )
    subreddit = reddit.subreddit('wallstreetbets')
    mentions = 0
    bullish = 0
    bearish = 0
    for post in subreddit.search(ticker, limit=50):
        mentions += 1
        sentiment = TextBlob(post.title + ' ' + post.selftext).sentiment.polarity
        if sentiment > 0.2:
            bullish += 1
        elif sentiment < -0.2:
            bearish += 1
    return {
        'mentions': mentions,
        'bullish': bullish,
        'bearish': bearish
    }

# Example usage:
if __name__ == "__main__":
    creds = {
        'client_id': 'zuIz-O-Mm5OLyU14WaRbFpoTHh01qw',
        'client_secret': 'YOUR_SECRET',
        'user_agent': 'aiautobottrader/0.1 by YOUR_USERNAME'
    }
    result = get_reddit_sentiment('AAPL', **creds)
    print(result)
