from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import numpy as np
# Load the Iris dataset
iris = load_iris()
X = iris.data
y = iris.target
# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
# Create Logistic Regression model
model = LogisticRegression(random_state=42, max_iter=200)
# Train the model using the training data
model.fit(X_train, y_train)
# Define function for predicting labels
def predict_label(raw_data):
    raw_data = np.array(raw_data).reshape(1, -1)
    probabilities = model.predict_proba(raw_data)
    return np.argmax(probabilities)