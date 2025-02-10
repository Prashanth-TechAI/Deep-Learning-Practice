import re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle

def preprocess_text(text):
    """ Clean and preprocess email text """
    text = text.lower()
    text = re.sub(r'\W+', ' ', text)  # Remove special characters
    return text

def extract_features(texts, vectorizer=None, train=False):
    """ Convert text data into numerical feature vectors using TF-IDF """
    if train:
        vectorizer = TfidfVectorizer(stop_words='english', max_features=500)
        features = vectorizer.fit_transform(texts).toarray()
        pickle.dump(vectorizer, open("vectorizer.pkl", "wb"))
        return features, vectorizer
    else:
        if not vectorizer:
            vectorizer = pickle.load(open("vectorizer.pkl", "rb"))
        return vectorizer.transform(texts).toarray()
