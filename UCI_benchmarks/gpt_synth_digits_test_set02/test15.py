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
# Initialize the Logistic Regression model
model = LogisticRegression(max_iter=10000)
# Fit the model using the training data
model.fit(X_train, y_train)
def predict_label(data):
    # Reshaping the data point to have the same format as the training data
    # We use -1 for unspecified value, it would then be inferred from the length of the array and remaining dimensions 
    data = np.array(data).reshape(1, -1)
    # Prediction probabilities for the data point
    probabilities = model.predict_proba(data)
    return probabilities[0]