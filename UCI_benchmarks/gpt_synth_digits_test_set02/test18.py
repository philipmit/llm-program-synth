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
model = LogisticRegression(max_iter=5000)
model.fit(X_train, y_train)
def predict_label(x_raw):
    if isinstance(x_raw, list):
        x_raw = np.array([np.array(xi) for xi in x_raw])
    elif isinstance(x_raw, tuple):
        x_raw = np.array([np.array(x_raw)]).reshape(-1, len(x_raw))
    return model.predict_proba(x_raw)