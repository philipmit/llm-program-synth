import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
# Load the digits dataset
digits = load_digits()
X = digits.data
y = digits.target
# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)
# Train a logistic regression model
model = LogisticRegression(max_iter=5000)
model.fit(X_train, y_train)
def predict_label(x_raw):
    # Ensure that the input data is 2D
    if len(x_raw.shape) == 1:
        x_raw = x_raw.reshape(1, -1)
    # Return the probabilities associated with the largest class
    return model.predict_proba(x_raw)[0]