from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import numpy as np
# Load the Breast Cancer Dataset
breast_cancer = load_breast_cancer()
X = breast_cancer.data
y = breast_cancer.target
# Feature scaling
scaler = StandardScaler()
X = scaler.fit_transform(X)
# Split the dataset into a training set and a testing set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)
# Define and train the Logistic Regression model
# Changed model to Logistic Regression with different hyperparameters for optimization
# Added penalty for regularization and 'newton-cg' solver for better convergence.
model = LogisticRegression(random_state=0, solver='newton-cg', penalty='l2', max_iter=10000)
model.fit(X_train, y_train)
def predict_label(raw_data):
    if raw_data.shape != (30,):
        raise ValueError('Invalid raw_data size. Expected (30,) but got', raw_data.shape)
    # make sure to scale the raw data as well
    processed_data = scaler.transform(raw_data.reshape(1, -1))
    prediction = model.predict_proba(processed_data)[:,1]
    return prediction[0]