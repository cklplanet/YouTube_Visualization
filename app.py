# app.py

from flask import Flask, render_template, request, redirect, url_for
import json
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from umap import UMAP
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import numpy as np

app = Flask(__name__)

# Default K value
DEFAULT_K = 3

# Function to perform clustering
def perform_clustering(data, k):
    # Embed the data with TF-IDF
    tfidf = TfidfVectorizer(sublinear_tf=True, min_df=5, norm='l2', encoding='latin-1', ngram_range=(1, 3), stop_words='english')

    data['concatenated'] = data.apply(lambda row: str(row['Titles']) +' '+ str(row['descriptions']) + str(row['captions']), axis=1)

    features_token = tfidf.fit_transform(data.concatenated).toarray()

    # Dimension reduction with UMAP
    umap = UMAP(n_components=2)
    umap_result = umap.fit_transform(features_token)

    # Cluster using K-means
    kmeans = KMeans(n_clusters=k)
    data['cluster'] = kmeans.fit_predict(umap_result)

    return data

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

        # Perform clustering with default K value
        df = perform_clustering(df, DEFAULT_K)
        app.df = df

        # Display the clustered groups on the results page
        return render_template('results.html', result=df, default_k=DEFAULT_K)

    return redirect(request.url)

# Define a route for handling the form submission with a new K value
@app.route('/update_k', methods=['POST'])
def update_k():
    new_k = int(request.form.get('new_k'))

    df = app.df
    df = perform_clustering(df, new_k)
    app.df = df

    # Display the clustered groups on the results page
    return render_template('results.html', result=df, default_k=new_k)

if __name__ == '__main__':
    app.run(debug=True)


