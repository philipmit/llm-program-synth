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
# Train logistic regression model
model = LogisticRegression(max_iter=10000)
model.fit(X_train, y_train)
def predict_label(sample):
    'Takes data for a single sample and predicts probabilities for each class'
     # Reshape sample to 2D array as model expects that shape
    sample_reshaped = np.array(sample).reshape(1, -1)
    # Predict probabilities
    probas = model.predict_proba(sample_reshaped)
    # Return the class label corresponding to the highest predicted probability
    return probas[0]