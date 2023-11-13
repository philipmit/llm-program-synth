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
# Standardize the dataset
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
# Define and train the MLP model 
clf = MLPClassifier(hidden_layer_sizes=(10,10,10), max_iter=500, random_state=42)
clf.fit(X_train_scaled, y_train)
def predict_label(raw_data):
    # Scale the raw data using the same scaler used for training
    scaled_data = scaler.transform([raw_data])
    # Predict probabilities with the trained MLP model
    probabilities = clf.predict_proba(scaled_data)
    return probabilities[0]
print(predict_label([5.1, 3.5, 1.4, 0.2]))