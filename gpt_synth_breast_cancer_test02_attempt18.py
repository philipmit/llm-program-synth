from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import numpy as np
# Load the Breast Cancer dataset
breast_cancer = load_breast_cancer()
X = breast_cancer.data
y = breast_cancer.target
# Standardize the features to potentially improve regression model performance
scaler = StandardScaler()
X = scaler.fit_transform(X)
# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)
# Defined Logisitic Regression model with balanced class weights and a higher number of iterations
# The 'balanced' mode uses the values of y to automatically adjust weights inversely proportional to class frequencies
# The 'liblinear’ solver is a good choice for small datasets and binary classification
model = LogisticRegression(class_weight='balanced', solver='liblinear', max_iter=10000)
# Train Logistic Regression model
model.fit(X_train, y_train)
def predict_label(sample):
    sample = scaler.transform(np.array(sample).reshape(1,-1))  # standardize sample
    probability = model.predict_proba(sample)[:,1]
    return probability[0]