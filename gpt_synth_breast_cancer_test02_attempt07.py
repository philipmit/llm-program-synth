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
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)
# Preprocessing: Standardization to make the algorithm much less sensitive to the scale of features
# Training a logistic regression model on the training set
# Increase the iterations to give more chance for the model to converge
# Use L2 regularization and tune C, inverse of regularization strength
model = make_pipeline(StandardScaler(), LogisticRegression(C=0.5, penalty='l2', max_iter=7000))
model.fit(X_train, y_train)
def predict_label(raw_data):
    # Raw data should be preprocessed before prediction
    raw_data = np.array(raw_data).reshape(1, -1)
    prediction = model.predict_proba(raw_data)
    return prediction[0][1]