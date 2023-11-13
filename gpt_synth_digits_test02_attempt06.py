from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import numpy as np
# Load the digits dataset
digits = load_digits()
X = digits.data
y = digits.target
# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)
# Normalize the data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
# Train a logistic regression model
clf = LogisticRegression(solver='liblinear', multi_class='auto')
clf.fit(X_train, y_train)
def predict_label(raw_data):
    processed_data = scaler.transform(raw_data.reshape(1, -1)) # reshape the data to 2D
    proba = clf.predict_proba(processed_data)
    return proba[0]  #Return the first (and only) element of the proba list