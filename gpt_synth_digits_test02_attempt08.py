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
# Train a logistic regression model
lr = LogisticRegression(max_iter=5000)
lr.fit(X_train, y_train)
def predict_label(raw_data):
    # Reshape and normalize raw_data
    normalized_data = np.array(raw_data).reshape(1, -1) / 16.0
    # Make prediction using the trained model
    pred_probs = lr.predict_proba(normalized_data)
    return pred_probs[0]