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
# Define and train the logistic regression model
lr = LogisticRegression(max_iter=1000)
lr.fit(X_train, y_train)
# Define the function predict_label
def predict_label(raw_data):
    # reshaping raw_data and returning first item of list to avoid nested lists
    return lr.predict_proba(np.array(raw_data).reshape(1,-1))[0]