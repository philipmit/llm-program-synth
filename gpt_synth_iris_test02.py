from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
# Load the Iris dataset
iris = load_iris()
X = iris.data
y = iris.target
# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Standardize the data
scaler = StandardScaler()
X_train_transformed = scaler.fit_transform(X_train)
# Train the multi-layer perceptron
mlp = MLPClassifier(hidden_layer_sizes=(10,), max_iter=1000, random_state=42)
mlp.fit(X_train_transformed, y_train)
def predict_label(x_raw_sample):
    # Preprocess the raw sample
    x_processed_sample = scaler.transform([x_raw_sample])
    # get the predicted probabilities
    predicted_probs = mlp.predict_proba(x_processed_sample)
    return predicted_probs