from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
# Load the digits dataset
digits = load_digits()
X = digits.data
y = digits.target
# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)
# Train the Logistic Regression model
model = LogisticRegression(max_iter=10000)
model.fit(X_train, y_train)
# Define a function called predict_label
def predict_label(sample):
    # Reshape the raw unprocessed data for a single sample
    reshaped_sample = sample.reshape(1, -1)
    # Use the trained model to predict the probabilities for that sample
    predicted_probabilities = model.predict_proba(reshaped_sample)
    # Return the predicted probabilities
    return predicted_probabilities[0]