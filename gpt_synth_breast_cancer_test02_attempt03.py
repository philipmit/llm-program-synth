from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import numpy as np
# Load the breast cancer dataset
breast_cancer = load_breast_cancer()
X = breast_cancer.data
Y = breast_cancer.target
# Split the dataset into a training set and a testing set
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
# Standardize the features to have a mean=0 and variance=1 
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
# Initialize and train the logistic regression model 
model = LogisticRegression(max_iter=500)
model.fit(X_train, y_train)
# Function to predict the label of a single sample
def predict_label(raw_sample):
    # remember to scale the new data as the model was trained with scaled data
    scaled_sample = scaler.transform(raw_sample.reshape(1, -1))
    prob = model.predict_proba(scaled_sample)
    return f"The probability of the sample being classified as 1 : {prob[0][1]}"