from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import numpy as np
# Load the Iris dataset
iris = load_iris()
X = iris.data
y = iris.target
# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
# Create logistic regression model
log_reg = LogisticRegression(max_iter=1000, random_state=42)
# Train the model
log_reg.fit(X_train, y_train)
def predict_label(data):
    """
    This function take a single sample, reshape it, and predict the probabilities of labels using the trained model.
    The input should be a 1D array-like object representing the features of a single sample.
    """
    data = np.array(data).reshape(1, -1)
    return log_reg.predict_proba(data)