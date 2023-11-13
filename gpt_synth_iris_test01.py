from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
# Load the Iris dataset
iris = load_iris()
X = iris.data
y = iris.target
# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Preprocess the data using StandardScaler
scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
# Define the multi-layer perceptron classifier
mlp = MLPClassifier(hidden_layer_sizes=(10,10,10), max_iter=1000)
# Train the model using the training data
mlp.fit(X_train, y_train)
# Define the prediction function
def predict_label(raw_data):
    processed_data = scaler.transform(raw_data.reshape(1, -1))
    predicted_probabilities = mlp.predict_proba(processed_data)
    return(predicted_probabilities)