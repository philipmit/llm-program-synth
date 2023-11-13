from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import numpy as np
# Load the Breast Cancer dataset
breast_cancer = load_breast_cancer()
X = breast_cancer.data
y = breast_cancer.target
# Standardize the features
scaler = StandardScaler()
X = scaler.fit_transform(X)
# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)
# Train a Logistic Regression model with a higher C value
log_reg = LogisticRegression(C=1.5, max_iter=10000)
log_reg.fit(X_train, y_train)
def predict_label(sample):
    # Normalize the sample using the same scaler fit on the training data
    sample = scaler.transform(np.array(sample).reshape(1, -1))
    # Return the probability of the label being 1
    prob = log_reg.predict_proba(sample)[0][1]
    return prob