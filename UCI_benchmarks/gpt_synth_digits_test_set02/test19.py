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
model = LogisticRegression(max_iter=5000)
model.fit(X_train, y_train)
# Function to predict the probabilities for a single sample
def predict_label(single_sample):
    # Using model.predict_proba to get the predicted probabilities
    predicted_probabilities = model.predict_proba(single_sample.reshape(1, -1))
    # Flatten the predicted probabilities to avoid creating a 3-dimensional array
    return predicted_probabilities.flatten()