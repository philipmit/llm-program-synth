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
# Create a logistic regression model and fit it to our training data
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)
def predict_label(x):    
    x = np.array(x).reshape(1, -1)  # Reshape the data
    return model.predict_proba(x)[0]   # Return the predicted probabilities directly