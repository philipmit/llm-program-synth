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
# Normalizing the features
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
# Define a Multi-layer Perceptron classifier
mlp = MLPClassifier(hidden_layer_sizes=(10, 10, 10), max_iter=1000)
# Train the Multi-layer Perceptron classifier
mlp.fit(X_train, y_train)
# Define the predicted_label function
def predict_label(raw_data):
    # Reshape the raw_data to fit the model input shape and normalize
    processed_data = sc_X.transform(raw_data.reshape(1, -1))
    predicted_proba = mlp.predict_proba(processed_data)
    return predicted_proba