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
# Define logistic regression model
model = LogisticRegression(max_iter=10000)
# Train the model on X_train and y_train
model.fit(X_train, y_train)
def predict_label(raw_data):
    # Model requires 2D array as input, so need to add an extra dimension
    reshaped_data = np.reshape(raw_data, (1, -1))
    # Predict probabilities
    probabilities = model.predict_proba(reshaped_data)
    return probabilities  