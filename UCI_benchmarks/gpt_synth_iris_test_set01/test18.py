from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
# Load the iris dataset
iris = load_iris()
X = iris.data
y = iris.target
# Split the dataset into a training set and a test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)
# Define a standard scaler
sc_X = StandardScaler()
# Fit and transform the training data
X_train = sc_X.fit_transform(X_train)
# Define an MLPClassifier
mlp = MLPClassifier(hidden_layer_sizes=(10, 10, 10), max_iter=1000)
# Train the model
mlp.fit(X_train, y_train)
# Define a function to predict the label of a given sample
def predict_label(raw_data):
    # Preprocess the data
    processed_data = sc_X.transform(raw_data.reshape(1, -1))
    # Predict the probabilities
    predicted_proba = mlp.predict_proba(processed_data)
    # Return the probabilities
    return predicted_proba[0]