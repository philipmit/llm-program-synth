from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
# Load the Iris dataset
iris = load_iris()
X = iris.data
y = iris.target
# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)
# Define the Logistic Regression model
model = LogisticRegression()
# Train the model using the training data
model.fit(X_train, y_train)
# Function to predict the label for a single sample
def predict_label(raw_data):
    # Reshape the raw data to match the model's expected input shape
    reshaped_data = raw_data.reshape(1, -1)
    # Get the predicted probabilities for the reshaped data
    predicted_probabilities = model.predict_proba(reshaped_data)
    return predicted_probabilities