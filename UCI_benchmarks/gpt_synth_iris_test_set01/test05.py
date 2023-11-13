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
# Standardizing the features
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
# Training a multi-layer perceptron
mlp = MLPClassifier(hidden_layer_sizes=(10,), max_iter=2000, random_state=42)
mlp.fit(X_train, y_train)
def predict_label(x):
    # Data pre-processing
    x = sc.transform([x])  # scale the sample using the same parameters used to scale the training data
    # Predict the probabilities of the classes
    proba = mlp.predict_proba(x)
    return proba
# Just as an example, let's see the predicted labels of all the training data.
for x in X_train[:5]:
    print(predict_label(x))