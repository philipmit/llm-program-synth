import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
# Load the Iris dataset
iris = load_iris()
X = iris.data
y = iris.target
# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Apply feature scaling to the dataset
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
# Define and train the MLP classifier
mlp = MLPClassifier(hidden_layer_sizes=(10, 10), max_iter=1000)
mlp.fit(X_train_scaled, y_train)
def predict_label(raw_data):
    # Ensure raw_data is 2D array. If it's 1D, convert it to 2D.
    if len(np.shape(raw_data)) == 1:
        raw_data = raw_data[np.newaxis, :]
    # Apply the same scaling to the raw_data
    raw_data_scaled = scaler.transform(raw_data)
    # Predict the probabilities
    predicted_probabilities = mlp.predict_proba(raw_data_scaled)
    # Since we are interested in the class with the highest probability, take argmax across columns
    predicted_class = np.argmax(predicted_probabilities, axis=1)
    # Return the result
    return predicted_class