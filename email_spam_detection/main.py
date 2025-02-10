import numpy as np
from perceptron import Perceptron
from preprocess import preprocess_text, extract_features
import pickle

# Load trained perceptron model
weights = np.load("weights.npy")
bias = np.load("bias.npy")

# Initialize perceptron
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))
perceptron = Perceptron(input_size=vectorizer.get_feature_names_out().shape[0])
perceptron.weights = weights
perceptron.bias = bias

def classify_email(email_text):
    """ Classify user-input email as spam or not spam """
    processed_text = preprocess_text(email_text)
    features = extract_features([processed_text], vectorizer=vectorizer)
    prediction = perceptron.predict(features[0])
    
    return "SPAM" if prediction == 1 else "NOT SPAM"

# Get user input and classify
if __name__ == "__main__":
    user_email = input("Enter an email message: ")
    result = classify_email(user_email)
    print(f"Prediction: {result}")
