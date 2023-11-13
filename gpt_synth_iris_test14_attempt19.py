from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
# Load the Iris dataset
iris = load_iris()
X = iris.data
y = iris.target
# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)
# Data normalization
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
# Create a Multi-Layer Perceptron classifier and train it
mlp = MLPClassifier(hidden_layer_sizes=(10,), activation='relu', solver='adam', max_iter=1000)
mlp.fit(X_train_scaled, y_train)
# Define the predict_label function
def predict_label(raw_sample):
    # Reshape raw_sample and normalize
    sample = raw_sample.reshape(1, -1)
    sample = scaler.transform(sample)
    # Use the trained MLP model to predict the probabilities
    predicted_probabilities = mlp.predict_proba(sample)
    # Flatten the predicted probabilities to be 1-D array
    predicted_probabilities = predicted_probabilities.flatten()
    return predicted_probabilities