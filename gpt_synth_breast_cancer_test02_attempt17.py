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
# Perform Logistic Regression
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)
def predict_label(raw_data):
  unprocessed_data = np.array(raw_data).reshape(1, -1)
  probability = model.predict_proba(unprocessed_data)
  return probability[0][1]