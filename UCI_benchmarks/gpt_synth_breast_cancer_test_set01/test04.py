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
# Scale the features to have zero mean and unit variance
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
# Train a logistic regression model
model = LogisticRegression()
model.fit(X_train, y_train)
def predict_label(raw_data):
    # Preprocess the raw_data using our scaler
    scaled_data = scaler.transform(raw_data.reshape(1, -1))
    # Predict the probability
    probability = model.predict_proba(scaled_data)
    # Return the probability of the label being 1
    return probability[0][1]