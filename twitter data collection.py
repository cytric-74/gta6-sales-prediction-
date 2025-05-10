import requests
import pandas as pd

BEARER_TOKEN = "need to get your x api key fam"

def fetch_twitter_data(query, max_results=50):
    headers = {"Authorization": f"Bearer {BEARER_TOKEN}"}
    url = f"https://api.twitter.com/2/tweets/search/recent?query={query}&tweet.fields=created_at,public_metrics&max_results={max_results}"
    response = requests.get(url, headers=headers)
    tweets = response.json()['data']
    
    df = pd.DataFrame([{
        'text': tweet['text'],
        'created_at': tweet['created_at'],
        'likes': tweet['public_metrics']['like_count']
    } for tweet in tweets])
    
    df.to_csv('data/twitter_data.csv', index=False)

fetch_twitter_data("GTA 6")
