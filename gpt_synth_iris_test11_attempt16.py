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
# We typically need to scale our data for neural networks
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
# Define the model
mlp = MLPClassifier(hidden_layer_sizes=(10,10), max_iter=1000)
# Train the model
mlp.fit(X_train_scaled, y_train)
def predict_label(raw_sample):
    # Preprocess the raw data sample
    raw_sample = np.array(raw_sample).reshape(1, -1)
    sample_scaled = scaler.transform(raw_sample)
    # Get the model to predict the probabilities of the classes
    sample_probs = mlp.predict_proba(sample_scaled)
    return sample_probs