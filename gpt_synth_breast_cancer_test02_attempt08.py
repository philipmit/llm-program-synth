from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
import numpy as np
# Load the Breast Cancer dataset
breast_cancer = load_breast_cancer()
X = breast_cancer.data
y = breast_cancer.target
# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Create logistic regression model with regularization and pipeline for preprocessing
log_reg = make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=10000))
# Train the model
log_reg.fit(X_train, y_train)
def predict_label(raw_data):
    # Convert input to 2D array
    raw_data = np.array(raw_data).reshape(1, -1)
    # Predict the probability of the sample being in class 1
    probability = log_reg.predict_proba(raw_data)[:, 1]
    return probability