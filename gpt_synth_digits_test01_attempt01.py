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
# Initialize the logistic Regression Model
logisticRegr = LogisticRegression(max_iter=10000)
# Train the model
logisticRegr.fit(X_train, y_train)
# Define the predict_label function
def predict_label(raw_data):
    return logisticRegr.predict_proba(np.reshape(raw_data, (1, -1)))