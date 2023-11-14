from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
import numpy as np
# Load the Iris dataset
iris = load_iris()
X = iris.data
y = iris.target
# Split the dataset into a training and a testing set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)
# Standardize features by removing the mean and scaling to unit variance
scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
# Train a multi-layer perceptron 
mlp = MLPClassifier(hidden_layer_sizes=(10, 10, 10), max_iter=1000, random_state=42)
mlp.fit(X_train, y_train)
def predict_label(raw_data):
    # Preprocess input data in the same way as the training data
    processed_data = scaler.transform(raw_data.reshape(1, -1))
    # Predict and return the class probabilities
    return mlp.predict_proba(processed_data)[0]