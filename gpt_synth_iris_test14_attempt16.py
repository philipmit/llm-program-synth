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
# Preprocessing the inputs
sc = StandardScaler()
sc.fit(X_train)
X_train_prep = sc.transform(X_train)
# Training multi-layer perceptron model
mlp = MLPClassifier(hidden_layer_sizes=(10,), max_iter=1000, random_state=42)
mlp.fit(X_train_prep, y_train)
# Defining the prediction function
def predict_label(X_raw):
    X_prep = sc.transform([X_raw])
    return mlp.predict_proba(X_prep)