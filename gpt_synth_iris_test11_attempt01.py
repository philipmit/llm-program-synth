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
# Normalize the data using StandardScaler
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
# Train the MLPClassifier with the Iris dataset
mlp = MLPClassifier(hidden_layer_sizes=(10, ), max_iter=1000, random_state=42)
mlp.fit(X_train, y_train)
def predict_label(X):
    # Assuming X is a raw data array, we first need to pre_process it
    X_processed = scaler.transform([X])
    predicted_proba = mlp.predict_proba(X_processed)
    return predicted_proba