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
# Initialize the LogisticRegression model
model = LogisticRegression(max_iter=200, random_state=42)
# Train the model using the training set
model.fit(X_train, y_train)
# Define a function to predict the class probabilities of a single sample
def predict_label(sample):
    # The model expects input as a 2D array so we need to reshape the single sample
    sample = np.array(sample).reshape(1, -1)
    # Return the predicted class probabilities for the sample
    return model.predict_proba(sample)[0]