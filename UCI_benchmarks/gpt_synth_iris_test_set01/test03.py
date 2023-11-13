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
# Normalize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
# Train the model
mlp = MLPClassifier(hidden_layer_sizes=(10,10,10), max_iter=1000, random_state=42)
mlp.fit(X_train, y_train)
def predict_label(x):
    # Ensure the input data is of shape (1, -1)
    if len(x.shape) == 1:
        x = x.reshape(1, -1)
    # Normalize the data
    x = scaler.transform(x)
    # Run prediction
    probabilities = mlp.predict_proba(x)
    return probabilities