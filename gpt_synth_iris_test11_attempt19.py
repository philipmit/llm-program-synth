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
# Scale the data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
# Train a multi-layer perceptron (neural network)
mlp = MLPClassifier(hidden_layer_sizes=(10, 10, 10), max_iter=1000)
mlp.fit(X_train, y_train)
def predict_label(raw_data):
    # Ensure raw_data is 1D
    if len(raw_data.shape) > 1:
        raw_data = np.squeeze(raw_data)
    # Scale the raw_data
    processed_data = scaler.transform([raw_data]) # Note the change here, considering the raw_data as single sample
    # Predict the probabilities
    probabilities = mlp.predict_proba(processed_data)
    return probabilities[0] # Returning the first element of the output, which is a 1D list