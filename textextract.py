from pypdf import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import numpy as np

def extract_text_from_pdf(pdf_path):
    pdf_reader = PdfReader(pdf_path)
    text = ''
    for page_num in range(len(pdf_reader.pages)):
        page = pdf_reader.pages[page_num]
        text += page.extract_text()
    return text

def preprocess_text(text):
    # Tokenize and remove stopwords
    stop_words = set(stopwords.words('english'))
    words = word_tokenize(text)
    filtered_words = [word.lower() for word in words if word.isalnum() and word.lower() not in stop_words]
    return ' '.join(filtered_words)

def extract_keywords_tfidf(text, top_n=10):
    tfidf = TfidfVectorizer(stop_words='english', ngram_range=(1, 2))

    features_token = tfidf.fit_transform([text]).toarray()
    title_ranking = np.sum(features_token, axis=0) / np.sum(features_token > 0, axis=0)
    indices = np.argsort(title_ranking)
    feature_names = np.array(tfidf.get_feature_names_out())[indices]
    unigrams = [v for v in feature_names if len(v.split(' ')) == 1]
    bigrams = [v for v in feature_names if len(v.split(' ')) == 2]
    unigrams_final = unigrams[-top_n:]
    bigrams_final = bigrams[-top_n:]
    unigrams_final.extend(bigrams_final)
    return unigrams_final

def extract_keywords_tfidf_multiple(documents, top_n=10):
    tfidf = TfidfVectorizer(stop_words='english', ngram_range=(1, 2))

    features_token = tfidf.fit_transform(documents).toarray()
    title_ranking = np.sum(features_token, axis=0) / np.sum(features_token > 0, axis=0)
    indices = np.argsort(title_ranking)
    feature_names = np.array(tfidf.get_feature_names_out())[indices]
    
    return feature_names[-top_n:]

def main(pdf_path):
    # Step 1: Extract text from PDF
    pdf_text = extract_text_from_pdf(pdf_path)
    
    # Step 2: Preprocess the text
    preprocessed_text = preprocess_text(pdf_text)
    
    # Step 3: Extract keywords using TF-IDF
    top_keywords_tfidf = extract_keywords_tfidf(preprocessed_text)
    
    # Print the top keywords
    print("Top 20 Keywords (TF-IDF):")
    print(top_keywords_tfidf)

if __name__ == "__main__":
    pdf_path = "Kaile_Chu_Essay_Example.pdf"  # Replace with the path to your PDF file
    main(pdf_path)
