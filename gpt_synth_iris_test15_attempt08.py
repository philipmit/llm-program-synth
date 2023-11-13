from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import numpy as np
# Load the Iris dataset
iris = load_iris()
X = iris.data
y = iris.target
# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
# Initiate Logistic Regression Model
model = LogisticRegression(multi_class='ovr', solver='liblinear')
# Train the model using the training sets
model.fit(X_train, y_train)
# Function to predict labels
def predict_label(input_data):
    input_data = np.array(input_data).reshape(1, -1)
    return model.predict_proba(input_data)