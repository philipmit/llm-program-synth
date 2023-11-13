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
# Create the logistic regression model
model = LogisticRegression(max_iter=200) # Increase max_iter if the model doesn't converge
# Train the model with the training data
model.fit(X_train, y_train)
def predict_label(single_sample):
    # The model expects a 2D array but we're providing a single sample
    # Therefore, we should convert this single sample into a 2D array
    single_sample = np.array(single_sample).reshape(1, -1)
    # Calculate the probabilities for each class
    predicted_proba = model.predict_proba(single_sample)
    return predicted_proba