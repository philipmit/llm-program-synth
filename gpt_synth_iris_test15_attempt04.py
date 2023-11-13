from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import numpy as np
# Load the Iris dataset
iris = load_iris()
X = iris.data
y = iris.target
# Split the dataset into training and testing sets.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
# Create a Logistic Regression model and train it
model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)
def predict_label(raw_data):
    # Ensure the input is in the correct format
    processed_data = np.array(raw_data).reshape(1,-1)
    # Make a prediction using the model
    prediction = model.predict_proba(processed_data)
    # Flatten the prediction array 
    flattened_prediction = prediction.flatten()
    # Return the predicted probabilities 
    return flattened_prediction