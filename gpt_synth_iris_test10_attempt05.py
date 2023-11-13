from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
import numpy as np
# Load the Iris dataset
iris = load_iris()
X = iris.data
y = iris.target
# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Initialize Multilayer Perceptron classifier
mlp = MLPClassifier(hidden_layer_sizes=(10,), max_iter=1000, random_state=42)
# Fit the data to the model
mlp.fit(X_train, y_train)
def predict_label(raw_data):
    """
    This function takes in raw unprocessed data for a single sample and 
    returns the predicted label of that sample using the MLP classifier.
    """
    # Preprocess raw_data to match the input shape
    raw_data = np.array(raw_data).reshape(1, -1)
    # Get the predicted probabilities for the sample
    y_pred_proba = mlp.predict_proba(raw_data)
    # Return the predicted probabilities
    return y_pred_proba[0]  # Adjust the return to be a 1D list.