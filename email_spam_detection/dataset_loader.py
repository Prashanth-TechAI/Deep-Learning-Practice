import pandas as pd
from preprocess import preprocess_text, extract_features

def load_dataset():
    """ Load spam email dataset """
    df = pd.read_csv("spam.csv", encoding='latin-1')
    df = df[['v1', 'v2']]
    df.columns = ['label', 'message']

    df['label'] = df['label'].map({'ham': 0, 'spam': 1})
    df['message'] = df['message'].apply(preprocess_text)

    X, vectorizer = extract_features(df['message'], train=True)
    y = df['label'].values
    return X, y, vectorizer
