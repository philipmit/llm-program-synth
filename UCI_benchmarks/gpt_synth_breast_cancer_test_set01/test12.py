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
# Scale the data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
# Train the model on the training data
log_reg = LogisticRegression(random_state=42)
log_reg.fit(X_train, y_train)
# Define the predict_label function
def predict_label(x):
    # Preprocess the data sample
    x = scaler.transform(np.array(x).reshape(1, -1))
    # Use the trained model to predict the label probability
    label_probability = log_reg.predict_proba(x)[0][1]
    return label_probability