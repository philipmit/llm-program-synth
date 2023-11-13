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
# Preprocessing: scale training and testing data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
# Define a multi-layer perceptron classifier
mlp = MLPClassifier(hidden_layer_sizes=(10,10,10), max_iter=1000)
# Training the model
mlp.fit(X_train, y_train)
def predict_label(raw_data):
    # Preprocess the data
    raw_data = scaler.transform(raw_data.reshape(1, -1))
    # Predict the probabilities
    predicted_probabilities = mlp.predict_proba(raw_data)
    return predicted_probabilities