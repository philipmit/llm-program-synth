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
# Scale the data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
# Define and train the model
model = MLPClassifier(hidden_layer_sizes=(10,10,10), max_iter=1000)
model.fit(X_train, y_train)
def predict_label(raw_data):
    # Scale the raw data
    scaled_data = scaler.transform([raw_data])
    # Predict the probabilities
    probabilities = model.predict_proba(scaled_data)
    return probabilities
# Example usage:
# note: insert raw data in the form of [sepal length, sepal width, petal length, petal width]
# raw_data = [5.1, 3.5, 1.4, 0.2]
# probabilities = predict_label(raw_data)
# print(probabilities)