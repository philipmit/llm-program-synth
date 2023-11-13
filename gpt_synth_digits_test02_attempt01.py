from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import numpy as np
# Load the digits dataset
digits = load_digits()
X = digits.data
y = digits.target
# Split the dataset into a training set and a test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)
# Create a logistic regression model
model = LogisticRegression(multi_class='ovr', solver='liblinear')
# Train the logistic regression model
model.fit(X_train, y_train)
def predict_label(x):
    # Reshape the input to be 2D as the model expects
    x = np.array(x).reshape(1, -1)
    # Return predicted probabilities and flatten 2D array to 1D
    return model.predict_proba(x).flatten()