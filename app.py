# app.py
from textextract import *

from flask import Flask, render_template, request, redirect, url_for
from flask_caching import Cache
from flask_sqlalchemy import SQLAlchemy
import json
import requests
import pandas as pd
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
from nltk.corpus import stopwords
import re
import string
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from umap import UMAP
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
try:
  from bertopic import BERTopic
except:
  from bertopic import BERTopic
from bertopic.representation import KeyBERTInspired
from youtube_transcript_api import YouTubeTranscriptApi
from hdbscan import HDBSCAN
from collections import Counter
import concurrent.futures
from sentence_transformers import SentenceTransformer

app = Flask(__name__)
cache = Cache(app, config={'CACHE_TYPE': 'simple'})

# Default K value
DEFAULT_K = 7

def get_cached_df():
    return None

def fetch_video_details(video_url):
    api_key = "YOUR_API_KEY"

    video_id = video_url.split("?v\u003d")[1]

    api_url = f"https://www.googleapis.com/youtube/v3/videos?id={video_id}&key={api_key}&part=snippet,statistics"
    response = requests.get(api_url)

    if response.status_code == 200:
        data = response.json()
        if 'items' in data and len(data['items']) > 0:
            video = data['items'][0]
            title = video['snippet']['title']
            description = video['snippet']['description']
            date = video['snippet']['publishedAt']
            thumbnail = video['snippet']['thumbnails']['high']['url']
            views = video['statistics']['viewCount']
            try:
                likes = video['statistics']['likeCount']
            except:
                likes = "N/A"
            return {'ID': video_id, 'title': title, 'description': description, 'date': date, 'thumbnail': thumbnail, 'views': views, 'likes': likes}

    return {'ID': video_id, 'title': 'Video not found', 'description': '', 'date': '', 'thumbnail': "", 'views': "", "likes": ""}

def extract_captions(ID):
    try:
        srt = YouTubeTranscriptApi.get_transcript(ID, ['en'])
        text = ''
        for i in srt:
            text += "{} ".format(i['text'])
        return text
    except:
        return " "

# old generate_dataset implementation
"""
def generate_dataset(video_data):
    videoID = []
    date = []
    titles = []
    descriptions = []
    captions = []
    counter = 0
    for video in video_data:
        if 'titleUrl' in video:
            video_details = fetch_video_details(video["titleUrl"])
            if video_details['title'] == "Video not found":
                continue
            videoID.append(video_details['ID'])
            date.append(video_details['date'])
            titles.append(video_details['title'])
            descriptions.append(video_details['description'])
            captions.append(extract_captions(video_details['ID']))
            counter += 1
            if counter >= LIMIT:
                break
    
    data = pd.DataFrame({'ID': videoID, 'Date': date, 'Titles': titles, 'descriptions': descriptions, 'captions':captions})
    return data
"""

def process_video(video):
    if 'titleUrl' in video:
        video_details = fetch_video_details(video["titleUrl"])
        if video_details['title'] == "Video not found":
            return None
        return {
            'ID': video_details['ID'],
            'Titles': video_details['title'],
            'descriptions': video_details['description'],
            'captions': extract_captions(video_details['ID']),
            'info': {'date':video['time'],"thumbnail":video_details['thumbnail'], "views":video_details['views'], "likes":video_details['likes']}
        }
    return None

def generate_dataset(video_data, limit):
    video_data = video_data[:limit]
    with concurrent.futures.ThreadPoolExecutor(max_workers=48) as executor:
        # Process each video concurrently
        processed_videos = list(executor.map(process_video, video_data))

    # Filter out None results
    processed_videos = [video for video in processed_videos if video is not None]

    # Create a DataFrame from the processed videos
    data = pd.DataFrame(processed_videos)
    
    return data

# Function to perform clustering
def perform_clustering(data, k, doc_keywords):
    # Embed the data with TF-IDF
    kmeans = KMeans(n_clusters=k, random_state=42)

    hdbscan_model = HDBSCAN(min_cluster_size=12, min_samples=12, metric='euclidean', cluster_selection_method='eom', prediction_data=True)
    n_components = 3
    umap_model = UMAP(n_components)
    representation_model = KeyBERTInspired()

    topic_model = BERTopic(hdbscan_model=kmeans, umap_model=umap_model, representation_model=representation_model)
    topics, probs = topic_model.fit_transform(data['documents'])

    vectorizer = TfidfVectorizer(sublinear_tf=True,  norm='l2', encoding='latin-1', ngram_range=(1, 3))
    

    candidates0 = doc_keywords

    groups = data.groupby(topic_model.topics_)
    total = pd.DataFrame()

    for i, group in groups:
        if i != -1:
            name = topic_model.get_topic_info()['Name'][i]
            keywords = topic_model.get_topic_info()['Representation'][i]
            group_members = group[['ID', 'info', 'Titles', 'descriptions']]
            column_mapping = {'ID': f'ID{i}', 'info': f'info{i}', 'Titles': f'title{i}', 'descriptions': f'description{i}'}
            keyword_data = {}

            combined_list = pd.concat([group['documents'], pd.Series(candidates0)], ignore_index=True).tolist()
            # Fit and transform the documents
            #if tfidf_key:
            tfidf_matrix = vectorizer.fit_transform(combined_list)
            embedding_combined = tfidf_matrix.toarray()
            #else:
                #embedding_combined = sbert_model_key.encode(combined_list, convert_to_tensor=True)
                #embedding_combined = embedding_combined.to('cpu').detach().numpy()

            doc_tfidf_embeddings = embedding_combined[:len(group['documents'])]
            keyword_tfidf_embeddings = embedding_combined[len(group['documents']):]

            doc_distances10_sub = cosine_similarity(doc_tfidf_embeddings, keyword_tfidf_embeddings)

            for i in range(len(keyword_tfidf_embeddings)):
                keyword_data[candidates0[i]] = {}
                subtopic_distance = doc_distances10_sub[:,i]
                subtopic_indices = [index+2 for index in subtopic_distance.argsort(axis=0)[::-1]]
                #print(f"Indexes for the subtopic {candidates0[i]}:")
                #print(subtopic_indexed)
                relevance_sum = np.sum(subtopic_distance)/len(subtopic_distance)
                keyword_data[candidates0[i]]["score"] = relevance_sum
                keyword_data[candidates0[i]]["indices"] = subtopic_indices


            new_row = {f'ID': keyword_data, f'info': keyword_data, f'Titles': keyword_data, f'descriptions': keyword_data}

            group_members = pd.concat([pd.DataFrame([new_row]), group_members], ignore_index=True)
            group_members = group_members.reset_index(drop=True)

            new_row0 = {f'ID': keywords, f'info': keywords, f'Titles': keywords, f'descriptions': keywords}

            group_members = pd.concat([pd.DataFrame([new_row0]), group_members], ignore_index=True)

            group_members = group_members.rename(columns=column_mapping)
            total = pd.concat([total, group_members], axis=1)

    # normalization based on MAX/min-max value for each keyword

    series = total.iloc[1]
    # Create a dictionary to store the total sum for each key
    max_values = {key: (max(entry[key]['score'] for entry in series)) for key in series.iloc[0].keys()}

    # Normalize each entry for each key
    normalized_series = {row_key: {key: {'score': (entry[key]['score'])/ (max_values[key]) if max_values[key] != 0 else 0, 'indices': entry[key]['indices']} for key in entry} for row_key, entry in series.items()}
    total.iloc[1] = normalized_series

    return total

# Define a route for the main page
@app.route('/')
def index():
    return render_template('index.html')

# Define a route for handling the JSON file upload
@app.route('/upload', methods=['POST'])
def upload():
    limit = int(request.form['entry_limit'])
    pdf_file = request.files['pdf_file']
    json_file = request.files['json_file']


    if pdf_file and json_file:
        # Read the PDF file
        # Step 1: Extract text from PDF
        pdf_text = extract_text_from_pdf(pdf_file)
        
        # Step 2: Preprocess the text
        preprocessed_text = preprocess_text(pdf_text)
        
        # Step 3: Extract keywords using TF-IDF
        top_keywords_tfidf = extract_keywords_tfidf(preprocessed_text)

        # Read the JSON file
        video_data = json.load(json_file)
        df = generate_dataset(video_data, limit)
        df['documents'] = df.apply(lambda row: str(row['Titles']) +' '+ str(row['descriptions']) + str(row['captions']), axis=1)
        
        app.df = df
        app.lst = top_keywords_tfidf
        #cache.set('cached_df', df, timeout=60*10)

        # Perform clustering with default K value
        df = perform_clustering(df, DEFAULT_K, top_keywords_tfidf)
        df.to_csv("actual_file_created_by_app_efficient.csv")
        df_csv = df.to_csv()

        # Display the clustered groups on the results page
        #return render_template('results.html', result=df_csv, default_k=DEFAULT_K)
        return render_template('results.html', result=df_csv, options_list=top_keywords_tfidf)

    return redirect(request.url)

# Define a route for handling the form submission with a new K value

@app.route('/update_k', methods=['POST'])
def update_k():
    new_k = int(request.form.get('new_k'))

    df = app.df
    top_keywords_tfidf = app.lst
    #df = cache.get('cached_df')
    df = perform_clustering(df, new_k)
    df_csv = df.to_csv()

    # Display the clustered groups on the results page
    return render_template('results.html', result=df_csv, options_list=top_keywords_tfidf)
    #return render_template('results.html', result=df_csv, default_k=new_k)


if __name__ == '__main__':
    app.run(debug=True)


