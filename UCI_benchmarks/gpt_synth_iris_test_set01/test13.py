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
# Standardize the data 
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
# Define the Multi-Layer Perceptron classifier and train
mlp = MLPClassifier(hidden_layer_sizes=(10, 10, 10), max_iter=1000)
mlp.fit(X_train_scaled, y_train)
# Define the predict_label function
def predict_label(raw_unprocessed_data):
    # Rescale the data and reshape to be compatible with the model
    raw_unprocessed_data = raw_unprocessed_data.reshape(1, -1)
    scaled_data = scaler.transform(raw_unprocessed_data)
    # Predict the label probabilities
    probabilities = mlp.predict_proba(scaled_data)
    # Since the result is a list of lists with a single element, we return the first element
    return probabilities[0]