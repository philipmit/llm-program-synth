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
# Train the logistic regression model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)
def predict_label(raw_data):
    """
    This function predict the probabilities of different classes for the given raw data.
    Parameters:
    -------------------
    raw_data: array-like of shape (n_features, )
        The input sample.
    Returns:
    -------------------
    probabilities : array-like of shape (n_classes,)
        Returns the probability of the sample for each class in the model.
    """
    raw_data = np.array(raw_data).reshape(1, -1)  # reshape input to 2D array
    probabilities = model.predict_proba(raw_data)[0]  # get the first (and only) prediction
    return probabilities