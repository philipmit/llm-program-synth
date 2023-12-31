from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import numpy as np
# Load the Iris dataset
iris = load_iris()
X = iris.data
y = iris.target
# Feature scaling
sc = StandardScaler()
X = sc.fit_transform(X)
# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)
# Create the logistic regression model
log_reg = LogisticRegression()
# Train the model with the training data
log_reg.fit(X_train, y_train)
def predict_label(raw_data):
    """Makes prediction using logistic regression model.
    Assumes raw_data is an array-like collection containing the features of one sample.
    Returns the probability of the predicted class.
    """
    # Rescaling raw_data with the predefined StandardScaler
    scaled_data = sc.transform([raw_data])
    # Get the array of predicted probabilities per class
    probabilities = log_reg.predict_proba(scaled_data)[0]
    # Normalize the predicted probabilities so that they sum up to 1 across all classes
    normalized_probabilities = probabilities / np.sum(probabilities)
    return normalized_probabilities