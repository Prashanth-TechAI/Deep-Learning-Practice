import numpy as np

class Perceptron:
    def __init__(self, input_size, learning_rate=0.01):
        self.weights = np.random.randn(input_size) * 0.01
        self.bias = 0
        self.learning_rate = learning_rate

    def activation_function(self, x):
        """ Step function for binary classification """
        return 1 if x > 0 else 0

    def predict(self, X):
        """ Forward pass: Compute output from inputs """
        weighted_sum = np.dot(X, self.weights) + self.bias
        return self.activation_function(weighted_sum)

    def train(self, X, y, epochs=50):
        """ Train the perceptron using supervised learning """
        for epoch in range(epochs):
            total_error = 0
            for xi, yi in zip(X, y):
                prediction = self.predict(xi)
                error = yi - prediction
                self.weights += self.learning_rate * error * xi
                self.bias += self.learning_rate * error
                total_error += abs(error)
            if epoch % 10 == 0:
                print(f"Epoch {epoch}, Error: {total_error}")
    