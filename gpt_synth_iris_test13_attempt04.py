from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
import numpy as np 
# Load the iris dataset
iris = load_iris()
X = iris.data
y = iris.target
# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)
# Standardize 
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
# Train the model
clf = MLPClassifier(hidden_layer_sizes=(100,), max_iter=300, random_state=42)
clf.fit(X_train_scaled, y_train)
# Define the predict_label function
def predict_label(raw_data):
    # Preprocess the input data 
    processed_data = scaler.transform(np.array(raw_data).reshape(1, -1))
    # Use the trained model to predict the probability 
    class_probs = clf.predict_proba(processed_data)
    return class_probs