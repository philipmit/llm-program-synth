from sklearn.neural_network import MLPClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
# Load the Iris dataset
iris = load_iris()
X = iris.data
y = iris.target
# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)
# Initialize a multi-layer perceptron model
mlp = MLPClassifier(hidden_layer_sizes=(10,), max_iter=1000, random_state=42)
# Fit (train) the model
mlp.fit(X_train, y_train)
def predict_label(raw_data):
    processed_data = mlp.predict_proba([raw_data])
    return processed_data