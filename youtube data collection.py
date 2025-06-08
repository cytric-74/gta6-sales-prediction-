from googleapiclient.discovery import build
import pandas as pd

YOUTUBE_API_KEY = 'need to get a youtube api key asap'

youtube = build('youtube', 'v3', developerKey=YOUTUBE_API_KEY)

def fetch_youtube_comments(query):
    search_response = youtube.search().list(
        q=query, part='snippet', type='video', maxResults=5
    ).execute()

    video_ids = [item['id']['videoId'] for item in search_response['items']]
    all_comments = []

    for vid in video_ids:
        comment_response = youtube.commentThreads().list(
            part='snippet', videoId=vid, textFormat='plainText', maxResults=50
        ).execute()

        for item in comment_response.get("items", []):
            comment = item['snippet']['topLevelComment']['snippet']['textDisplay']
            all_comments.append(comment)

    df = pd.DataFrame({'comment': all_comments})
    df.to_csv('data/youtube_data.csv', index=False)
    return df