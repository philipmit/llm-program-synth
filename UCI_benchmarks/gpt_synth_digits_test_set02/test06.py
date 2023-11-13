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
# Train a logistic regression model on the training set
model = LogisticRegression(max_iter=5000)
model.fit(X_train, y_train)
def predict_label(sample):
    # Reshape the sample into 2D array, then predict
    sample_processed = sample.reshape(1, -1)
    return model.predict_proba(sample_processed)[0]