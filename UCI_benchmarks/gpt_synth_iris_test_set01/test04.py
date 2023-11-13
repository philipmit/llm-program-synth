from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
# Load the Iris dataset
iris = load_iris()
X = iris.data
y = iris.target
# Splitting the dataset into the training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)
# Preprocessing: scale training data using standard scaler
scaler = StandardScaler().fit(X_train) 
X_train = scaler.transform(X_train)
# Defining a multi-layer perceptron classifier
mlp = MLPClassifier(hidden_layer_sizes=(10,10,10), max_iter=1000)
# Training the model
mlp.fit(X_train, y_train)
# Define a function to scale data and predict labels
def predict_label(raw_data):
    # Preprocess the raw data
    raw_data = scaler.transform(raw_data.reshape(1, -1))
    # Predict the probabilities
    predicted_probabilities = mlp.predict_proba(raw_data)
    return predicted_probabilities[0] # return flat list