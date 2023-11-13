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
# Scaling the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
# Train the Multi-layer Perceptron
clf = MLPClassifier(hidden_layer_sizes=(10,), max_iter=1000, random_state=42)
clf.fit(X_train, y_train)
def predict_label(raw_data):
    # Transform raw_data using the previously declared scaler
    raw_data = scaler.transform(raw_data.reshape(1, -1))
    # Return the predicted probabilities for each class
    prediction = clf.predict_proba(raw_data)
    # Getting the 1st element of prediction to flatten it
    return prediction[0]