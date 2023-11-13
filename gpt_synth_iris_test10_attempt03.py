from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier 
from sklearn.preprocessing import StandardScaler
import numpy as np
# Load the Iris dataset
iris = load_iris()
X = iris.data
y = iris.target
# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Initialize a Standard Scaler and fit_transform on the training data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
# Initialize the MLP Classifier and fit it to the scaled training data
mlp = MLPClassifier(hidden_layer_sizes=(10,), activation='relu', solver='adam', max_iter=500)
mlp.fit(X_train_scaled, y_train)
def predict_label(single_sample):
    """
    This function receives a single sample (a numpy array of shape (4,)), scales it according to the scaler trained on the training set,
    predicts the class probabilities using the trained MLP classifier, and returns these probabilities.
    """
    single_sample = single_sample.reshape(1, -1)  # Reshape the single sample
    scaled_sample = scaler.transform(single_sample)  # Scale using the previously fitted scaler
    predicted_probabilities = mlp.predict_proba(scaled_sample)  # Predict the class probabilities
    return predicted_probabilities[0]