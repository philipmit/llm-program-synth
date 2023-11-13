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
# Normalize the Features using Standard Scaler
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
# Train a Multi-layer Perceptron classifier
mlp = MLPClassifier(hidden_layer_sizes=(10,), max_iter=1000, random_state=42)
mlp.fit(X_train, y_train)
def predict_label(X_raw):
    # Ensuring the input is transformed exactly as during the training
    X_processed = scaler.transform([X_raw])
    # Predict and return probabilities
    y_prob = mlp.predict_proba(X_processed)
    return y_prob[0]