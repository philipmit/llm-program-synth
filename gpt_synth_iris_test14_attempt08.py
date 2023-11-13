# Required Libraries
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
# Preprocess the data
scaler = StandardScaler()
# Fit the scaler on the training data, and transform the training data
X_train_scaled = scaler.fit_transform(X_train)
# Initialize the MLP Classifier
mlp = MLPClassifier(hidden_layer_sizes=(100,), max_iter=300, random_state=42)
# Train the model
mlp.fit(X_train_scaled, y_train)
def predict_label(data):
    # Preprocessing the single sample
    data_scaled = scaler.transform(data.reshape(1, -1))
    # Predict and return the probabilities
    return mlp.predict_proba(data_scaled)[0]