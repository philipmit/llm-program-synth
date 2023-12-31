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
# Train the model using Logistic Regression
log_reg = LogisticRegression(max_iter = 5000)
log_reg.fit(X_train, y_train)
# Define the predict_label function
def predict_label(input_data):
    input_data = np.array(input_data).reshape(1,-1) #reshape the input_data to 2D as required by the predict_proba
    probabilities = log_reg.predict_proba(input_data) #find the probabilities of each class
    return probabilities.ravel()  # Converting the 2D array into a 1D array