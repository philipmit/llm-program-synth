from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import numpy as np
# Load the wine dataset
wine = load_wine()
X = wine.data
y = wine.target
# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)
# Train a logistic regression model
model = LogisticRegression(max_iter=10000)
model.fit(X_train, y_train)
def predict_label(raw_unprocessed_data):
    processed_data = np.array(raw_unprocessed_data).reshape(1, -1)
    probabilities = model.predict_proba(processed_data)
    return probabilities.flatten()