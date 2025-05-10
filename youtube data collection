from googleapiclient.discovery import build
import pandas as pd

API_KEY = 'need to get a youtube api key asap'
youtube = build('youtube', 'v3', developerKey=API_KEY)

def get_youtube_comments(query):
    request = youtube.search().list(q=query, part='snippet', type='video', maxResults=10)
    videos = request.execute()
    
    video_ids = [item['id']['videoId'] for item in videos['items']]
    all_comments = []

    for vid in video_ids:
        comments = youtube.commentThreads().list(
            part='snippet', videoId=vid, textFormat='plainText', maxResults=50
        ).execute()
        
        for item in comments.get("items", []):
            all_comments.append(item['snippet']['topLevelComment']['snippet']['textDisplay'])
    
    df = pd.DataFrame({'comment': all_comments})
    df.to_csv('data/youtube_data.csv', index=False)

get_youtube_comments("GTA 6 trailer")
