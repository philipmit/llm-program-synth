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
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)
# Data normalization
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
# Define the MLP model
model = MLPClassifier(hidden_layer_sizes=(100,50), max_iter=1000)
model.fit(X_train, y_train)
def predict_label(sample):
    sample = scaler.transform([sample])  # apply the same transformation to the sample
    probs = model.predict_proba(sample)  # obtain the probabilities from the model
    return probs[0]