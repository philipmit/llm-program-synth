from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
# Load the digits dataset
digits = load_digits()
X = digits.data
y = digits.target
# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)
# Create logistic regression model
model = LogisticRegression(max_iter=10000)
# Fit the model to the training data
model.fit(X_train, y_train)
def predict_label(raw_data):
    raw_data = data.reshape(1, -1) 
    proba = model.predict_proba(raw_data)
    return proba[0]