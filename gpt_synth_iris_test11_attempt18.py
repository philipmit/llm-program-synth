from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
# Load the Iris dataset
iris = load_iris()
X = iris.data
y = iris.target
# Scale the data 
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
# Initialize the multilayer perceptron classifier
mlp = MLPClassifier(hidden_layer_sizes=(10, 10, 10), max_iter=1000)
# Train the multilayer perceptron classifier
mlp.fit(X_train, y_train)
def predict_label(unprocessed_data):
    # Preprocess data using the same scaler fit on the training data
    processed_data = scaler.transform(unprocessed_data.reshape(1, -1))
    # Predict the label
    proba = mlp.predict_proba(processed_data)
    return proba