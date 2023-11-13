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
# Initialize the standard scaler and fit on the training data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
# Initialize Multi-Layer Perceptron Classifier.
mlp = MLPClassifier(hidden_layer_sizes=(10, 10, 10),
                    max_iter=1000,
                    alpha=0.0001,
                    solver='sgd',
                    verbose=10,
                    random_state=21,
                    tol=0.000000001)
# Train the Model
mlp.fit(X_train_scaled, y_train)
def predict_label(raw_sample):
    # Check if data is 1-dimensional
    if len(raw_sample.shape) == 1:
        # Reshape the raw unprocessed input data to be 2-dimensional.
        raw_sample = np.reshape(raw_sample, (1, -1))
    # Preprocess the raw unprocessed input data
    raw_sample_scaled = scaler.transform(raw_sample)
    # Predict and return the probabilities for the input data
    return mlp.predict_proba(raw_sample_scaled)[0]