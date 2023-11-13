from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import numpy as np
# Load the wine dataset
wine = load_wine()
X = wine.data
y = wine.target
# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)
# Standardize the features to have mean=0 and variance=1
scaler = StandardScaler().fit(X_train)
# Apply the scaler to the split training set
X_train = scaler.transform(X_train)
# Define and train the logistic regression model
lr = LogisticRegression(max_iter=1000).fit(X_train, y_train)
def predict_label(raw_data):
    """
    Given the raw data of a sample, predict the probabilities for each class using the trained model.
    The raw data should be an array of shape (13, ) and should contain the values for the 13 features. 
    The order of the features should match the order of the features in the Wine dataset.
    Returns an array of shape (3, ) containing the predicted probabilities for each of the three classes.
    """
    # Preprocess the raw data using the preprocessor
    input_data = scaler.transform(raw_data.reshape(1, -1))
    # Use the logistic regression model to predict the probabilities
    probabilities = lr.predict_proba(input_data)[0]
    return probabilities