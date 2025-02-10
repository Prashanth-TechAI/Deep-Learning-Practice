from perceptron import Perceptron
from dataset_loader import load_dataset
from sklearn.model_selection import train_test_split
import numpy as np

# Load dataset
X, y, vectorizer = load_dataset()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize perceptron
perceptron = Perceptron(input_size=X.shape[1], learning_rate=0.01)

# Train the model
perceptron.train(X_train, y_train, epochs=100)

# Save trained model
np.save("weights.npy", perceptron.weights)
np.save("bias.npy", perceptron.bias)

print("Training complete. Model saved.")
