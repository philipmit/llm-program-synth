from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
import numpy as np
# Load the Iris dataset
iris = load_iris()
X = iris.data
y = iris.target
# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Define the multi-layer perceptron model
mlp = MLPClassifier(hidden_layer_sizes=(10, ), max_iter=1000, random_state=42)
# Train the model using training data
mlp.fit(X_train, y_train)
def predict_label(input_data):
    # Ensure the input data is a 2D array
    if input_data.ndim == 1:
        input_data = np.array([input_data])
    # Return predicted probabilities for a single sample
    return mlp.predict_proba(input_data)[0]