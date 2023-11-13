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
logisticRegr = LogisticRegression(max_iter=4000)
# Train the logistic regression model
logisticRegr.fit(X_train, y_train)
# Define the prediction function
def predict_label(sample_data):
    sample_data_reshaped = sample_data.reshape(1, -1)
    # Note: the output of logisticRegr.predict_proba is a 2D array, where the first dimension corresponds
    # to the number of samples and the second dimension corresponds to the number of classes.
    # Since we've reshaped the sample data to be a single sample, we take the first element of the output to
    # return a 1D array of class probabilities.
    return logisticRegr.predict_proba(sample_data_reshaped)[0]