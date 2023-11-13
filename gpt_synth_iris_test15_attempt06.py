from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import numpy as np
# Load the Iris dataset
iris = load_iris()
X = iris.data
y = iris.target
# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
# Initialize and train the logistic regression model
model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)
def predict_label(raw_data):
    # Reshape the array to 2D if it's 1D 
    if len(raw_data.shape) == 1:
        raw_data = np.reshape(raw_data, (1, -1))
    # Use the logistic regression model to predict the probabilites of each class
    probabilities = model.predict_proba(raw_data)
    return probabilities