"""
Twitter Sentiment Fetcher for Trading Bot
-----------------------------------------
Uses Tweepy to fetch tweets for ticker mentions
Analyzes sentiment using TextBlob
Requires Twitter API credentials in .env
"""

import tweepy
from textblob import TextBlob
import os
from dotenv import load_dotenv

load_dotenv()

TWITTER_API_KEY: str = os.getenv('TWITTER_API_KEY')
TWITTER_API_KEY_SECRET: str = os.getenv('TWITTER_API_KEY_SECRET')
TWITTER_ACCESS_TOKEN: str = os.getenv('TWITTER_ACCESS_TOKEN')
TWITTER_ACCESS_TOKEN_SECRET: str = os.getenv('TWITTER_ACCESS_TOKEN_SECRET')


def get_twitter_sentiment(
    ticker: str,
    consumer_key: str,
    consumer_secret: str,
    access_token: str,
    access_token_secret: str
) -> dict:
    """
    Fetch tweets for a ticker and analyze sentiment.
    Returns a dict with mentions, bullish, and bearish counts.
    """
    try:
        auth = tweepy.OAuth1UserHandler(
            consumer_key, consumer_secret, access_token, access_token_secret
        )
        api = tweepy.API(auth)
        mentions = 0
        bullish = 0
        bearish = 0
        for tweet in tweepy.Cursor(api.search_tweets, q=ticker, lang='en', tweet_mode='extended').items(50):
            mentions += 1
            text = getattr(tweet, 'full_text', getattr(tweet, 'text', ''))
            sentiment = TextBlob(text).sentiment.polarity
            if sentiment > 0.2:
                bullish += 1
            elif sentiment < -0.2:
                bearish += 1
        return {
            'mentions': mentions,
            'bullish': bullish,
            'bearish': bearish
        }
    except Exception as e:
        print(f"Error fetching Twitter sentiment: {e}")
        return {'mentions': 0, 'bullish': 0, 'bearish': 0}


# Example usage:
if __name__ == "__main__":
    result = get_twitter_sentiment(
        'AAPL',
        TWITTER_API_KEY,
        TWITTER_API_KEY_SECRET,
        TWITTER_ACCESS_TOKEN,
        TWITTER_ACCESS_TOKEN_SECRET
    )
    print(result)
