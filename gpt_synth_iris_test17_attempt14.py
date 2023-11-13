from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
# Load the Iris dataset
iris = load_iris()
X = iris.data
y = iris.target
# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)
# Define the logistic regression model
model = LogisticRegression()
# Train the model
model.fit(X_train, y_train)
def predict_label(raw_data):
    # Ensure the raw_data is reshaped appropriately for the model
    if len(raw_data.shape) == 1:
        raw_data = raw_data.reshape(1, -1)
    # Use the logistic regression model to predict the probability of each class
    predicted_probability = model.predict_proba(raw_data)
    return predicted_probability[0]