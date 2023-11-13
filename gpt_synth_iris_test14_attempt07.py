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
# Normalize the data
scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
# Define the MLP model
model = MLPClassifier(hidden_layer_sizes=(10,10), max_iter=500, random_state=42)
# Train the MLP model
model.fit(X_train, y_train)
def predict_label(sample):
    # Normalize the sample
    sample = scaler.transform([sample])
    # Predict label probabilities
    return model.predict_proba(sample)