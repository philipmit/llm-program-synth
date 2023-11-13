from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
# Load the digits dataset
digits = load_digits()
X = digits.data
y = digits.target
# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)
# Create Logistic Regression model
model = LogisticRegression(max_iter=5000)
# Train the model
model.fit(X_train, y_train)
def predict_label(sample_data):
    # Predict the probabilities for the sample data
    probabilities = model.predict_proba([sample_data])
    # The function is expected to return the probabilities
    return probabilities[0]