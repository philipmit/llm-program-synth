from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import numpy as np
# Load the Iris dataset
iris = load_iris()
X = iris.data
y = iris.target
# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)
# Train a logistic regression model
model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)
def predict_label(raw_data):
    """
    This function receives raw input data for a single sample and 
    returns the predicted probabilities for that sample as computed by the already trained logistic regression model.
    """
    # Reshape raw_data to a 2D array as model.predict_proba needs a 2D array as input
    raw_data_reshaped = np.array(raw_data).reshape(1, -1)
    # Get probabilities from the logistic regression model
    prediction_proba = model.predict_proba(raw_data_reshaped)
    # Return the probabilities for all classes flattened to 1D
    return prediction_proba.flatten()