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
# Train the logistic regression model
log_reg = LogisticRegression(max_iter=5000)
log_reg.fit(X_train, y_train)
def predict_label(raw_data):
    # Reshape raw_data
    reshaped_data = np.array(raw_data).reshape(1, -1)
    # Predict probabilities
    prediction_result = log_reg.predict_proba(reshaped_data)
    return prediction_result[0]