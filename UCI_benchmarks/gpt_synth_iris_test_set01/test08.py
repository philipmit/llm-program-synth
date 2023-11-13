import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
iris = load_iris()
X = iris.data
y = iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)
# Initialize StandardScaler
scaler = StandardScaler()
scaler.fit(X_train)
# Apply transform to both the training set and the test set
X_train_std = scaler.transform(X_train)
# Create an instance of MLPClassifier
mlp = MLPClassifier(hidden_layer_sizes=(10,10,10),max_iter=1000)
mlp.fit(X_train_std, y_train)
def predict_label(raw_data):
    scaled_data = scaler.transform(raw_data.reshape(1, -1))
    # Retrive the label with highest probability
    return np.argmax(mlp.predict_proba(scaled_data), axis=1)