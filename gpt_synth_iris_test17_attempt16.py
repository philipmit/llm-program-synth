from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import numpy as np
# Load the Iris dataset
iris = load_iris()
X = iris.data
y = iris.target
# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)
# Create and train the Logistic Regression model
lr = LogisticRegression(max_iter=200)
lr.fit(X_train, y_train)
def predict_label(sample):
    # Reshape sample to match the shape required by sklearn models (n_samples, n_features)
    sample = np.array(sample).reshape(1, -1)
    # Predict probabilities
    probas = lr.predict_proba(sample)
    # Return probabilities as a 1d list
    return probas.ravel()