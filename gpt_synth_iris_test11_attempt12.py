from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
import numpy as np
# Load the Iris dataset
iris = load_iris()
X = iris.data
y = iris.target
# Split the dataset into a training set and a testing set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Scale the features for better performance
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
# Train a multi-layer perceptron with default parameters
mlp = MLPClassifier(random_state=42)
mlp.fit(X_train, y_train)
# Define the function to predict the label of a single sample
def predict_label(single_sample):
    # Adjust the single sample to fit the input shape
    single_sample = np.array(single_sample).reshape(1, -1)
    # Scale the single sample's features like the training set
    single_sample = scaler.transform(single_sample)
    # Return the predicted probabilities of the classes
    return mlp.predict_proba(single_sample)