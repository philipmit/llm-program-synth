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
# Data Preprocessing - Scale the inputs
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
# Create a multi-layer perceptron classifier and train it on the training data
mlp = MLPClassifier(hidden_layer_sizes=(10, 10, 10), max_iter=1000)
mlp.fit(X_train, y_train)
def predict_label(single_sample):
    """Predict probabilities for a single sample."""
    # Standardize the sample and reshape it into a 2D array
    single_sample_scaled = sc.transform(single_sample.reshape(1, -1))
    # Use the trained model to predict the probabilities for each class
    predicted_probabilities = mlp.predict_proba(single_sample_scaled)
    return predicted_probabilities