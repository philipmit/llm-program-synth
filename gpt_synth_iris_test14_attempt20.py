from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
import numpy as np
# Load the Iris dataset
iris = load_iris()
X = iris.data
y = iris.target
# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)
# Data normalization
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
# Create a multi-layer perceptron classifier
mlp = MLPClassifier(hidden_layer_sizes=(10, 10, 10), max_iter=1000)
# Train the model
mlp.fit(X_train_scaled, y_train)
# predict_label function
def predict_label(raw_data):
    # Reshape the data and transform using the previously fitted scaler
    processed_data = scaler.transform(raw_data.reshape(1, -1))
    # Get the class probabilities
    probabilities = mlp.predict_proba(processed_data)
    return probabilities[0]