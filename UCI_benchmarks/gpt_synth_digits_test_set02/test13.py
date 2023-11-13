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
# Instantiate the model
logreg = LogisticRegression(max_iter=10000)
# Fit the model on the training data
logreg.fit(X_train, y_train)
def predict_label(raw_data):
    # Reshape the raw data to match the shape the model expects
    raw_data = np.array(raw_data).reshape(1, -1)
    # Predict the probabilities of each class
    predicted_probabilities = logreg.predict_proba(raw_data)
    # Get the index with the maximum probability
    predicted_label = np.argmax(predicted_probabilities)
    return predicted_label