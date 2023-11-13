from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
import numpy as np
# Load the Iris dataset
iris = load_iris()
X = iris.data
y = iris.target
# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Preprocessing: Standardize the dataset
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
# Training
clf = MLPClassifier(hidden_layer_sizes=(10,), max_iter=1000, random_state=42)
clf.fit(X_train, y_train)
def predict_label(unprocessed_data):
    # Preprocess the single sample
    scale_unprocessed_data = scaler.transform([unprocessed_data])
    # Get the prediction probabilities.
    predicted_probabilities = clf.predict_proba(scale_unprocessed_data)
    return predicted_probabilities.flatten()