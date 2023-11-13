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
# Initialize and fit the logistic regression model
lr = LogisticRegression(max_iter=10000)
lr.fit(X_train, y_train)
def predict_label(raw_data):
    # Reshape the data to have shape (1, -1)
    reshaped_data = np.array(raw_data).reshape(1, -1)
    # Return the probability of each class
    # We use ravel() function to return contiguous flattened array(1D array with all the input-array elements)
    return lr.predict_proba(reshaped_data).ravel()