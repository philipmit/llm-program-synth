from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import numpy as np
# Load the Iris dataset
iris = load_iris()
X = iris.data
y = iris.target
# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Standardize the features
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
# Define and train the multi-layer perceptron
mlp = MLPClassifier(hidden_layer_sizes=(10,10), max_iter=1000, probability=True)
mlp.fit(X_train_s, y_train)
def predict_label(raw_data):
    scaled_data = scaler.transform(np.array(raw_data).reshape(1, -1))
    probability = mlp.predict_proba(scaled_data)[0]
    return probability