from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
# Load the Iris dataset
iris = load_iris()
X = iris.data
y = iris.target
# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)
# Feature scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
# Create Multi-layer Perceptron classifier
mlp = MLPClassifier(hidden_layer_sizes=(10, 10, 10), max_iter=1000)
# Train the classifier
mlp.fit(X_train, y_train)
def predict_label(raw_data):
    # Transform raw_data with the same scaler used in training
    scaled_data = scaler.transform([raw_data])
    # Predict label
    proba = mlp.predict_proba(scaled_data)
    # Remove unwanted extra dimension
    proba = proba[0]
    return proba