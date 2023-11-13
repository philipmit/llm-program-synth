from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
import numpy as np
# Load the Iris dataset
iris = load_iris()
X = iris.data
y = iris.target
# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Scale the features of the dataset
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
# Train the model
model = MLPClassifier(hidden_layer_sizes=(10, 10, 10), max_iter=1000)
model.fit(X_train_scaled, y_train)
def predict_label(raw_data):
    # Reshaping raw_data from 1D array to 2D array for the scaler
    raw_data = np.array(raw_data).reshape(1, -1) 
    # Scale the raw data
    scaled_data = scaler.transform(raw_data)
    # Predict probabilities
    probabilities = model.predict_proba(scaled_data)
    return probabilities