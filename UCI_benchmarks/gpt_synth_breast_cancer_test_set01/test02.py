from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import numpy as np
# Load the Breast Cancer dataset
breast_cancer = load_breast_cancer()
X = breast_cancer.data
y = breast_cancer.target
# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)
# Define and train the Logistic Regression model
model = LogisticRegression(max_iter=10000)
model.fit(X_train, y_train)
def predict_label(raw_data):
    if raw_data.shape != (30,):
        raise ValueError('Invalid raw_data size. Expected (30,) but got', raw_data.shape)
    processed_data = raw_data.reshape(1, -1)
    prediction = model.predict_proba(processed_data)[:,1]
    return prediction[0]