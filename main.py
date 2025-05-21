import pandas as pd
from sklearn.linear_model import LinearRegression

def combine_and_predict():
    tw = pd.read_csv("data/twitter_data_sentiment.csv")["sentiment"]
    yt = pd.read_csv("data/youtube_data_sentiment.csv")["sentiment"]
    nw = pd.read_csv("data/news_articles_sentiment.csv")["sentiment"]

    avg_sentiment = pd.DataFrame({
        "twitter": [tw.mean()],
        "youtube": [yt.mean()],
        "news": [nw.mean()]
    })

    # Dummy model (train on historic sales if available)
    model = LinearRegression()
    X = avg_sentiment
    y = [50] # assuming sales
    model.fit(X, y)
    
    pred = model.predict(X)
    print(f"Predicted Sales (in millions): {pred[0]}")

combine_and_predict()
