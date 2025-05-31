import os
import requests
import pandas as pd
from bs4 import BeautifulSoup
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
import numpy as np
import time
from datetime import datetime
import logging
from typing import Optional, Dict, List, Union

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('data/sentiment_analysis.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Constants
DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)
analyzer = SentimentIntensityAnalyzer()

# Configuration class (consider moving sensitive data to environment variables)
class Config:
    TWITTER_BEARER_TOKEN = os.getenv("TWITTER_BEARER_TOKEN", "YOUR_TWITTER_BEARER_TOKEN")
    YOUTUBE_API_KEY = os.getenv("YOUTUBE_API_KEY", "YOUR_YOUTUBE_API_KEY")
    REQUEST_TIMEOUT = 30
    MAX_RETRIES = 3
    RETRY_DELAY = 2

# Helper functions
def safe_request(url: str, headers: Optional[Dict] = None, params: Optional[Dict] = None, 
                max_retries: int = Config.MAX_RETRIES) -> Optional[requests.Response]:
    """Make HTTP requests with error handling and retries."""
    for attempt in range(max_retries):
        try:
            response = requests.get(
                url, 
                headers=headers, 
                params=params, 
                timeout=Config.REQUEST_TIMEOUT
            )
            response.raise_for_status()
            return response
        except requests.exceptions.RequestException as e:
            logger.warning(f"Attempt {attempt + 1} failed for {url}: {str(e)}")
            if attempt < max_retries - 1:
                time.sleep(Config.RETRY_DELAY * (attempt + 1))
            continue
    logger.error(f"Failed to fetch data from {url} after {max_retries} attempts")
    return None

def save_data(df: pd.DataFrame, filename: str) -> bool:
    """Save DataFrame to CSV with error handling."""
    try:
        filepath = os.path.join(DATA_DIR, filename)
        df.to_csv(filepath, index=False)
        logger.info(f"Data successfully saved to {filepath}")
        return True
    except Exception as e:
        logger.error(f"Failed to save data to {filename}: {str(e)}")
        return False

# Twitter data collection
def fetch_twitter_data(query: str, max_results: int = 50) -> Optional[pd.DataFrame]:
    """Fetch recent tweets containing the query."""
    headers = {"Authorization": f"Bearer {Config.TWITTER_BEARER_TOKEN}"}
    url = "https://api.twitter.com/2/tweets/search/recent"
    params = {
        "query": query,
        "tweet.fields": "created_at,public_metrics",
        "max_results": max_results
    }
    
    response = safe_request(url, headers=headers, params=params)
    if not response:
        return None
    
    try:
        data = response.json()
        tweets = data.get('data', [])
        
        if not tweets:
            logger.warning("No tweets found for the given query")
            return pd.DataFrame()
            
        df = pd.DataFrame([{
            'text': tweet.get('text', ''),
            'created_at': tweet.get('created_at', ''),
            'likes': tweet['public_metrics'].get('like_count', 0) if 'public_metrics' in tweet else 0
        } for tweet in tweets])
        
        if not save_data(df, 'twitter_data.csv'):
            return None
            
        return df
    except Exception as e:
        logger.error(f"Error processing Twitter data: {str(e)}")
        return None

# YouTube data collection
def fetch_youtube_comments(query: str, max_videos: int = 5, max_comments: int = 50) -> Optional[pd.DataFrame]:
    """Fetch comments from YouTube videos matching the query."""
    try:
        youtube = build('youtube', 'v3', developerKey=Config.YOUTUBE_API_KEY)
    except Exception as e:
        logger.error(f"Failed to initialize YouTube API: {str(e)}")
        return None

    try:
        search_response = youtube.search().list(
            q=query, part='snippet', type='video', maxResults=max_videos
        ).execute()
    except HttpError as e:
        logger.error(f"YouTube API search error: {str(e)}")
        return None

    video_ids = [item['id']['videoId'] for item in search_response.get('items', [])]
    all_comments = []

    for vid in video_ids:
        try:
            comment_response = youtube.commentThreads().list(
                part='snippet', 
                videoId=vid, 
                textFormat='plainText', 
                maxResults=max_comments
            ).execute()
        except HttpError as e:
            logger.warning(f"Failed to fetch comments for video {vid}: {str(e)}")
            continue

        for item in comment_response.get("items", []):
            comment = item['snippet']['topLevelComment']['snippet'].get('textDisplay', '')
            all_comments.append(comment)

    df = pd.DataFrame({'comment': all_comments})
    
    if not save_data(df, 'youtube_data.csv'):
        return None
        
    return df

# News scraping
def scrape_news(query: str) -> Optional[pd.DataFrame]:
    """Scrape news articles from Google News."""
    formatted_query = query.replace(" ", "%20")
    search_url = f"https://news.google.com/search?q={formatted_query}&hl=en-US&gl=US&ceid=US:en"
    
    response = safe_request(search_url)
    if not response:
        return None
    
    try:
        soup = BeautifulSoup(response.text, 'html.parser')
        articles = []
        
        for a in soup.select('article h3'):
            if a.a:
                title = a.text.strip()
                link = "https://news.google.com" + a.a['href'][1:]
                articles.append({'title': title, 'link': link})

        df = pd.DataFrame(articles)
        
        if not save_data(df, 'news_articles.csv'):
            return None
            
        return df
    except Exception as e:
        logger.error(f"Error parsing news data: {str(e)}")
        return None

# Sentiment analysis
def analyze_sentiment(df: pd.DataFrame, column: str) -> Optional[pd.DataFrame]:
    """Add sentiment analysis to DataFrame."""
    if df.empty:
        logger.warning("Empty DataFrame received for sentiment analysis")
        return df
        
    try:
        df['sentiment'] = df[column].astype(str).apply(
            lambda x: analyzer.polarity_scores(x)['compound']
        )
        return df
    except Exception as e:
        logger.error(f"Error in sentiment analysis: {str(e)}")
        return None

# Prediction models
def combine_and_predict() -> Optional[pd.DataFrame]:
    """Combine data sources and predict sales."""
    try:
        # Load and analyze data
        twitter_df = analyze_sentiment(pd.read_csv("data/twitter_data.csv"), "text")
        youtube_df = analyze_sentiment(pd.read_csv("data/youtube_data.csv"), "comment")
        news_df = analyze_sentiment(pd.read_csv("data/news_articles.csv"), "title")

        if twitter_df is None or youtube_df is None or news_df is None:
            logger.error("Failed to load one or more data sources")
            return None

        # Calculate average sentiments
        avg_sentiments = pd.DataFrame({
            'twitter_sentiment': [twitter_df['sentiment'].mean()],
            'youtube_sentiment': [youtube_df['sentiment'].mean()],
            'news_sentiment': [news_df['sentiment'].mean()],
            'timestamp': [datetime.now().isoformat()]
        })

        # Generate training data (simulated)
        np.random.seed(42)
        X_train = pd.DataFrame({
            'twitter_sentiment': np.random.uniform(-1, 1, 100),
            'youtube_sentiment': np.random.uniform(-1, 1, 100),
            'news_sentiment': np.random.uniform(-1, 1, 100),
        })
        y_train = X_train.mean(axis=1) * 50 + 85  # Mocked-up sales range

        # Initialize and train models
        models = {
            'Linear Regression': LinearRegression(),
            'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
            'MLP Regressor': MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=500, random_state=42)
        }

        predictions = {}
        for name, model in models.items():
            try:
                model.fit(X_train, y_train)
                pred = model.predict(avg_sentiments[['twitter_sentiment', 'youtube_sentiment', 'news_sentiment']])[0]
                predictions[name] = round(pred, 2)
                logger.info(f"{name} Predicted GTA 6 Sales (in millions): {round(pred, 2)}")
            except Exception as e:
                logger.error(f"Error in {name} prediction: {str(e)}")
                predictions[name] = None

        # Save results
        result_df = avg_sentiments.copy()
        for name, pred in predictions.items():
            result_df[name.replace(" ", "_").lower() + '_prediction'] = pred

        if not save_data(result_df, "sales_prediction.csv"):
            return None
            
        return result_df

    except Exception as e:
        logger.error(f"Error in prediction pipeline: {str(e)}")
        return None

def main():
    """Main execution function."""
    logger.info("Starting sentiment analysis pipeline...")
    
    logger.info("Collecting Twitter data...")
    twitter_data = fetch_twitter_data("GTA 6")
    if twitter_data is None:
        logger.error("Twitter data collection failed")
    
    logger.info("Collecting YouTube comments...")
    youtube_data = fetch_youtube_comments("GTA 6 trailer")
    if youtube_data is None:
        logger.error("YouTube data collection failed")
    
    logger.info("Scraping news articles...")
    news_data = scrape_news("GTA 6")
    if news_data is None:
        logger.error("News data collection failed")
    
    logger.info("Analyzing sentiment and predicting sales with multiple models...")
    predictions = combine_and_predict()
    
    if predictions is not None:
        logger.info("All tasks completed successfully!")
        logger.info("Prediction results:\n%s", predictions.to_string())
    else:
        logger.error("Pipeline completed with errors")
    
    logger.info("Check the 'data/' folder for output files")

if __name__ == "__main__":
    main()