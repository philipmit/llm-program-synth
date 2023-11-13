import numpy as np
from sklearn.datasets import load_iris
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
# Load the Iris dataset
iris = load_iris()
X = iris.data
y = iris.target
# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)
# Standardize the features to help training the neural network
scaler = StandardScaler()
# Fit only on the training data
scaler.fit(X_train)
# Apply transformation to the training and the testing set
X_train = scaler.transform(X_train)
# Initialize the multi-layer perceptron classifier
mlp = MLPClassifier(hidden_layer_sizes=(10,10), max_iter=1000, alpha=1e-4, solver='sgd', verbose=10, tol=1e-4, random_state=1, learning_rate_init=.1)
# Train the model
mlp.fit(X_train, y_train)
# Define the function to predict labels
def predict_label(x):
    # Make sure that x is 2D
    x = np.array(x).reshape(1, -1)
    # Scale the features
    x = scaler.transform(x)
    # Use the trained model to predict the class probabilities and retrieve the results for the first instance
    return mlp.predict_proba(x)[0]