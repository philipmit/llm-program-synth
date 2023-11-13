from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
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
    # Expect input data to be a NumPy array. 
    # Reshape the input data to be a 2D array with 1 row if it's a 1D array
    if len(input_data.shape) == 1:
        input_data = input_data.reshape(1, -1)
    # The model will output the predicted_probabilities which will be a 2D array
    predicted_probabilities = model.predict_proba(input_data)
    # Return the predicted probabilities as a list
    return list(predicted_probabilities[0])