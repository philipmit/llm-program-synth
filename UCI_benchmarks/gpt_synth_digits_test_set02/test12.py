from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import numpy as np
# Load the digits dataset
digits = load_digits()
X = digits.data
y = digits.target
# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)
# Train a logistic regression model
model = LogisticRegression(max_iter=10000)
model.fit(X_train, y_train)
def predict_label(sample):
    """
    This function takes raw unprocessed data for a single sample 
    and returns the predicted probabilities for the 10 possible classes for that sample.
    """
    sample = np.array(sample).reshape(1, -1)
    return model.predict_proba(sample)