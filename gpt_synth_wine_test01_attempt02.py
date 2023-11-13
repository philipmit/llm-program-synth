from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
import numpy as np
# Load the wine dataset
wine = load_wine()
X = wine.data
y = wine.target
# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)
# Train a Logistic Regression model
lr = LogisticRegression(max_iter=1000)
lr.fit(X_train, y_train)
def predict_label(data):
    # As logistic regression predicts for multiple samples simultaneously, 
    # We put our single instance into an array in order to feed it to predict_proba
    data_reshaped = np.array(data).reshape(1, -1)
    class_prediction = lr.predict(data_reshaped)
    return class_prediction