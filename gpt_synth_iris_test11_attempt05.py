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
# We also use a StandardScaler to ensure our MLP performs well
scaler = StandardScaler()
# Fit only to the training data
scaler.fit(X_train)
# Now apply the transformations to the data
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)
# Initialize the MLPClassifier
mlp = MLPClassifier(hidden_layer_sizes=(10, 10, 10), max_iter=1000)
# Fit the data
mlp.fit(X_train, y_train)
# Here we define the predict_label function
def predict_label(sample):
    # As our model is trained on scaled data, we should also scale the input
    sample = scaler.transform([sample])
    # Probabilities prediction 
    probabilities = mlp.predict_proba(sample)[0]
    return probabilities