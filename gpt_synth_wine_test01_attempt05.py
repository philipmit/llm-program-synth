import numpy as np
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
# Load the wine dataset
wine = load_wine()
X = wine.data
y = wine.target
# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)
# Standardize the features
scaler = StandardScaler().fit(X_train)
X_train = scaler.transform(X_train)
# Train the logistic regression model
model = LogisticRegression(max_iter=10000)
model.fit(X_train, y_train)
# Define the prediction function
def predict_label(x_raw):
    # Transform the raw input to 2D array
    x_raw = np.array(x_raw).reshape(1, -1)
    # Standardize the raw input
    x_std = scaler.transform(x_raw)
    # Predict the probabilities and return the result
    probas = model.predict_proba(x_std)
    return probas[0] # Return the first element of prediction as a 1D array instead of a 2D array