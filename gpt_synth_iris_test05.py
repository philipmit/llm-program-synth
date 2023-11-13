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
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Data normalization
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
# Training a multi-layer perceptron
mlp = MLPClassifier(hidden_layer_sizes=(10,10,), max_iter=1000)
mlp.fit(X_train_scaled, y_train)
def predict_label(sample):
    # Ensure the sample is reshaped correctly and then scale it
    sample = np.array(sample).reshape(1, -1)
    sample_scaled = scaler.transform(sample)
    # Predict probabilities and return
    probabilities = mlp.predict_proba(sample_scaled)
    return probabilities