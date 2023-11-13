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
# Scale the data to mean=0 and variance=1
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
# Define the MLP model and train it
model = MLPClassifier(hidden_layer_sizes=(10,10,10), max_iter=1000)
model.fit(X_train, y_train)
# Function to predict labels
def predict_label(X):
    X = scaler.transform([X])
    return model.predict_proba(X)