# Set up
import streamlit as st
from googleapiclient.discovery import build
import pandas as pd
import numpy as np
import string
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import pickle
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer

# Lấy api key từ youtube
api_key = 'AIzaSyCJHKxYYjvIX-xs5wxxouiylhQKYyytROM'

# Load model
model = pickle.load(open('model.pkl', 'rb'))
preprocessor = joblib.load('preprocessor.pkl')

# Lấy data từ video
def getVideoDetails(video_id):
    youtube = build('youtube', 'v3', developerKey=api_key)
    res = youtube.videos().list(part='snippet,statistics',id=video_id).execute()
    
    if 'items' in res and len(res['items']) > 0:
        video_info = res['items'][0]
        video_details = {
            'title': video_info['snippet']['title'],
            'description': video_info['snippet']['description'],
            'published_at': video_info['snippet']['publishedAt'],
            'view_count': video_info['statistics']['viewCount'],
            'like_count': video_info['statistics']['likeCount']
        }
        return video_details
    else:
        return None
    
def getComments(video_id):
    youtube = build('youtube', 'v3', developerKey=api_key)
    res = youtube.commentThreads().list(part='snippet',videoId=video_id).execute()

    comments = []
    for item in res['items']:
        # Thu thập data
        date = item['snippet']['topLevelComment']['snippet']['publishedAt']
        user = item['snippet']['topLevelComment']['snippet']['authorDisplayName']

        comment = item['snippet']['topLevelComment']['snippet']['textDisplay']
        likeCount = item['snippet']['topLevelComment']['snippet']['likeCount']
        comments.append([date, user, comment, likeCount])
    return comments

def preProcess(text):
    message = []
    for x in text:
        if x in string.punctuation:
            continue
        message.append(x)
    message = ''.join(message)

    stop_words = set(stopwords.words('english'))
    words= []
    for x in message.split():
        if x.lower() in stop_words:
            continue
        words.append(x)

    lemmatizer = WordNetLemmatizer()
    lemma = ' '.join([lemmatizer.lemmatize(x.lower()) for x in words])
    return lemma

def prediction(text):
    input = [text, len(text)]
    input = preprocessor.transform([input[0]])
    predict = model.predict(input)

    if predict[0] == 0:
        return 'No spam'
    else: return 'Spam'

# Inteface
st.title("YouTube Comments Scraper")
url = st.text_input("Enter YouTube Video URL:")

if st.button('Get'):
    try:
        video_id = url.split('v=')[1]
        data = getVideoDetails(video_id)
        commments = getComments(video_id)
        
        if data:
            st.write("### Video Details:")
            st.write(f"**Title:** {data['title']}")
            st.write(f"**Description:** {data['description']}")
            st.write(f"**Published At:** {data['published_at']}")
            st.write(f"**View Count:** {data['view_count']}")
            st.write(f"**Like Count:** {data['like_count']}")
        
        if commments:
            df_comment = pd.DataFrame(commments, columns=['data', 'user_id', 'content','like_count'])
            df_comment['clean_content'] = df_comment['content'].apply(preProcess)
            df_comment['label'] = df_comment['content'].apply(prediction)
            st.table(df_comment)
    except Exception as e:
        st.write(e)
        # st.write("Error: Invalid YouTube URL or API limit reached.")