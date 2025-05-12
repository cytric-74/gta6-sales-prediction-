import requests
from bs4 import BeautifulSoup
import pandas as pd

def scrape_news(query):   # main keyword used for searching in google search engine
    base_url = "https://news.google.com/search"
    search_url = f"{base_url}?q={query}"
    response = requests.get(search_url)
    soup = BeautifulSoup(response.text, 'html.parser') 

    articles = []   # making an array to store the data from the article
    for a in soup.select('article h3'):
        if a.a:
            title = a.text
            link = "https://news.google.com" + a.a['href'][1:]
            articles.append({'title': title, 'link': link})
    
    df = pd.DataFrame(articles)
    df.to_csv('data/news_articles.csv', index=False)

scrape_news("GTA 6")