from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import numpy as np
# Load the Iris dataset
iris = load_iris()
X = iris.data
y = iris.target
# Split the dataset into training and testing 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
# Train the logistic regression model
model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)
def predict_label(input_data):
    # Reshape the input to match the format the model requires
    reshaped_data = input_data.reshape(1, -1)
    probabilities = model.predict_proba(reshaped_data)
    # Get the index of the maximum probability
    max_prob_index = np.argmax(probabilities)
    # Return the maximum probability value
    return probabilities[0][max_prob_index]