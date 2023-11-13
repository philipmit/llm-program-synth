from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import numpy as np
# Load the wine dataset
wine = load_wine()
X = wine.data
y = wine.target
# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)
# Normalize features for better performance by logistic regression
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
# Train a logistic regression model on the training set
log_reg = LogisticRegression(random_state=42)
log_reg.fit(X_train, y_train)
# Define the function to predict probabilities for a new sample
def predict_label(raw_sample):
    # Preprocess the input using the same scaler used for training
    preprocess_sample = scaler.transform(np.array(raw_sample).reshape(1, -1))
    # Use the trained model to predict probabilities
    probabilities = log_reg.predict_proba(preprocess_sample)
    return probabilities.flatten() # Ensure that the output is a 1D array