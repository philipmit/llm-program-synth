from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import numpy as np
# Load the Iris dataset
iris = load_iris()
X = iris.data
y = iris.target
# Split the dataset into a training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)
# Train the Logistic Regression model
model = LogisticRegression(random_state=42, max_iter=1000)
model.fit(X_train, y_train)
# Define the prediction function
def predict_label(raw_data):
    # Ensure the raw_data is in the correct format
    data = np.array(raw_data).reshape(1, -1)
    # Use our trained model to predict the probabilities
    predictions = model.predict_proba(data)
    return predictions