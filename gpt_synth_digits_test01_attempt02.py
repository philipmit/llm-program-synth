from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import numpy as np
# Load the digits dataset
digits = load_digits()
X = digits.data
y = digits.target
# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)
# Training the Logistic Regression model
model = LogisticRegression(max_iter=10000)
model.fit(X_train, y_train)
def predict_label(single_sample_data):
    single_sample_data = np.array(single_sample_data).reshape(1, -1)
    # Get the predicted probabilities and flatten the array to 1 dimension
    pred_probs = model.predict_proba(single_sample_data).flatten()
    return pred_probs