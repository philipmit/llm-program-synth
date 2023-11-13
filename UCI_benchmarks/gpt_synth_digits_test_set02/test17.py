from sklearn.datasets import load_digits
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import numpy as np
# Load the digits dataset
digits = load_digits()
X = digits.data
y = digits.target
# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)
# Create a logistic regression object
logisticRegr = LogisticRegression(max_iter=10000)
# Train the model
logisticRegr.fit(X_train, y_train)
def predict_label(raw_unprocessed_data):
    # Reshaping raw_unprocessed_data to a 2D array for prediction
    raw_unprocessed_data = np.array(raw_unprocessed_data).reshape(1, -1)
    # Return predicted probabilities
    return logisticRegr.predict_proba(raw_unprocessed_data)