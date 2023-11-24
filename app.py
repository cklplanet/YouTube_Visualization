# app.py

from flask import Flask, render_template, request, redirect, url_for
import json
import pandas as pd
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
from hdbscan import HDBSCAN

app = Flask(__name__)

# Default K value
DEFAULT_K = 7

# Function to perform clustering
def perform_clustering(data, k):
    # Embed the data with TF-IDF
    kmeans = KMeans(n_clusters=k, random_state=42)

    hdbscan_model = HDBSCAN(min_cluster_size=12, min_samples=12, metric='euclidean', cluster_selection_method='eom', prediction_data=True)
    n_components = 3
    umap_model = UMAP(n_components)
    representation_model = KeyBERTInspired()

    topic_model = BERTopic(hdbscan_model=kmeans, umap_model=umap_model, representation_model=representation_model)
    topics, probs = topic_model.fit_transform(data['documents'])

    vectorizer = TfidfVectorizer(sublinear_tf=True,  norm='l2', encoding='latin-1', ngram_range=(1, 3))
    

    candidates0 = ["Jujutsu Kaisen", "manga", "anime series", "Gege Akutami", "cursed spirits", "Yuji Itadori", "Jujutsu Sorcery", "cursed energy", "Tokyo Metropolitan Jujutsu Technical High School", "Megumi Fushiguro", "Nobara Kugisaki", "Satoru Gojo", "life and death", "friendship", "justice", "popularity", "critical acclaim", "awards", "Sugoi Japan Award", "Japanese pop culture"]

    groups = data.groupby(topic_model.topics_)
    total = pd.DataFrame()

    for i, group in groups:
        name = topic_model.get_topic_info()['Name'][i]
        keywords = topic_model.get_topic_info()['Representation'][i]
        group_members = group[['ID', 'Date']]
        column_mapping = {'ID': f'ID{i}', 'Date': f'Date{i}'}
        keyword_data = {}

        combined_list = pd.concat([group['documents'], pd.Series(candidates0)], ignore_index=True).tolist()
        # Fit and transform the documents
        tfidf_matrix = vectorizer.fit_transform(combined_list)

        embedding_combined = tfidf_matrix.toarray()

        doc_tfidf_embeddings = embedding_combined[:len(group['documents'])]
        keyword_tfidf_embeddings = embedding_combined[len(group['documents']):]

        doc_distances10_sub = cosine_similarity(doc_tfidf_embeddings, keyword_tfidf_embeddings)

        for j in range(len(keyword_tfidf_embeddings)):
            keyword_data[candidates0[j]] = {}
            subtopic_distance = doc_distances10_sub[:,j]
            subtopic_indices = [index+1 for index in subtopic_distance.argsort(axis=0)[::-1]]
            #print(f"Indexes for the subtopic {candidates0[i]}:")
            #print(subtopic_indexed)
            relevance_sum = np.sum(subtopic_distance)
            keyword_data[candidates0[j]]["score"] = relevance_sum
            keyword_data[candidates0[j]]["indices"] = subtopic_indices


        #new_row = {f'ID': keyword_data, f'Date': keyword_data}

        #group_members = pd.concat([pd.DataFrame([new_row]), group_members], ignore_index=True)
        #group_members = group_members.reset_index(drop=True)

        #new_row0 = {f'ID': keywords, f'Date': keywords}
        new_row0 = {f'ID': name, f'Date': keywords}

        group_members = pd.concat([pd.DataFrame([new_row0]), group_members], ignore_index=True)

        group_members = group_members.rename(columns=column_mapping)
        total = pd.concat([total, group_members], axis=1)

    #total.to_csv(f"clusters.csv")

    return total

# Define a route for the main page
@app.route('/')
def index():
    return render_template('index.html')

# Define a route for handling the JSON file upload
@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return redirect(request.url)

    file = request.files['file']

    if file.filename == '':
        return redirect(request.url)

    if file and file.filename.endswith('.csv'):
        # Read the CSV file
        df = pd.read_csv(file)
        df['documents'] = df.apply(lambda row: str(row['Titles']) +' '+ str(row['descriptions']) + str(row['captions']), axis=1)
        app.df = df

        # Perform clustering with default K value
        df = perform_clustering(df, DEFAULT_K)
        df_csv = df.to_csv()

        # Display the clustered groups on the results page
        return render_template('results.html', result=df_csv, default_k=DEFAULT_K)

    return redirect(request.url)

# Define a route for handling the form submission with a new K value
@app.route('/update_k', methods=['POST'])
def update_k():
    new_k = int(request.form.get('new_k'))

    df = app.df
    df = perform_clustering(df, new_k)
    df_csv = df.to_csv()

    # Display the clustered groups on the results page
    return render_template('results.html', result=df_csv, default_k=new_k)

if __name__ == '__main__':
    app.run(debug=True)


