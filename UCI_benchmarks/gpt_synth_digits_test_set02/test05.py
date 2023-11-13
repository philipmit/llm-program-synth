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
# Train a Logistic Regression model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)
def predict_label(sample):
    # Make sure the sample is 1D
    if len(sample.shape) != 1:
        raise ValueError(f"Expected a 1D sample, but got a {len(sample.shape)}D sample")
    # Reshape the sample to 2D (n_samples, n_features)
    sample = sample.reshape(1, -1)
    # Predict the probabilities of each class
    probabilities = model.predict_proba(sample)
    return probabilities[0]