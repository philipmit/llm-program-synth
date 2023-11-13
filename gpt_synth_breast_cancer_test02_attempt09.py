from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
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
# Train the Logistic Regression model
model = LogisticRegression()
model.fit(X_train, y_train)
def predict_label(sample):
    sample = np.array(sample).reshape(1, -1)
    sample = scaler.transform(sample)
    probability = model.predict_proba(sample)[0,1]
    return probability