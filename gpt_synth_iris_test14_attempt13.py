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
# Standardize features by removing the mean and scaling to unit variance
scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
# Defining the Multi-Layer Perceptron Classifier
mlp = MLPClassifier(hidden_layer_sizes=(10, 10, 10), max_iter=1000)
# Training the model
mlp.fit(X_train, y_train)
def predict_label(raw_data):
    # Preprocessing the raw_data using the same scaler used on training data
    raw_scaled = scaler.transform([raw_data])
    predicted_probs = mlp.predict_proba(raw_scaled)
    # Return only the first array of predictions probabilities as we are only predicting for one instance here.
    # Removes an extra dimension from the output.
    return predicted_probs[0]