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
# Scaling the features for better results
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
# Instantiate the Multi Layer Perceptron Classifier
mlp = MLPClassifier(hidden_layer_sizes=(10, 10, 10), max_iter=1000)
# Train the model using the training data
mlp.fit(X_train, y_train)
def predict_label(sample):
    # Preprocess the sample by scaling it
    sample_scaled = scaler.transform([sample])
    # Use the model to predict the probability of each class
    probabilities = mlp.predict_proba(sample_scaled)
    return probabilities[0]
# As per the restrictions, a test of predict_label is not included here.