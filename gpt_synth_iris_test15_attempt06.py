from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import numpy as np
iris = load_iris()
X = iris.data
y = iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)
def predict_label(raw_data):
    if len(raw_data.shape) == 1:
        raw_data = np.reshape(raw_data, (1, -1))
    probabilities = model.predict_proba(raw_data)
    #Return the predicted probabilities for each class:
    return probabilities