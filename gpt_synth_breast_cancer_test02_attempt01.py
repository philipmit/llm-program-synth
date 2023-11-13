from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import numpy as np
# Load the Breast Cancer dataset
breast_cancer = load_breast_cancer()
X = breast_cancer.data
y = breast_cancer.target
# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)
# Standardize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
# Train a Logistic Regression Model
model = LogisticRegression()
model.fit(X_train, y_train)
# Define a function to predict the label of a new sample
def predict_label(raw_data):
    # Process the raw data and predict the probability of the label being 1
    processed_data = scaler.transform(raw_data.reshape(1, -1))
    probability = model.predict_proba(processed_data)[0][1]
    return probability