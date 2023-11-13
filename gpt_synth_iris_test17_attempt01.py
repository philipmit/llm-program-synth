from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
# Load the Iris dataset
iris = load_iris()
X = iris.data
y = iris.target
# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)
# Fit a logistic regression model to the data
model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)
# Define a function to make predictions on a single sample
def predict_label(sample_input):
    sample_input_reshaped = sample_input.reshape(1, -1)  # reshape single sample for prediction
    prediction=model.predict_proba(sample_input_reshaped)
    return prediction[0] #return as a 1-D array