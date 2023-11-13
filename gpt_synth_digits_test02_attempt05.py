from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
# Load the digits dataset
digits = load_digits()
X = digits.data
y = digits.target
# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)
# Define logistic regression model
model = LogisticRegression(max_iter=3000)
# Train the model using our training data
model.fit(X_train, y_train)
# Define a function that takes raw data and returns predicted probabilities
def predict_label(raw_data):
    # The model.predict_proba method returns a 2D array, 
    # but since we're only predicting for one observation, we only need the first element of the result
    return model.predict_proba(raw_data.reshape(1, -1))[0]