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
# Initialize the Multi-Layer Perceptron Classifier
model = MLPClassifier(hidden_layer_sizes=(10,10,10), max_iter=1000)
# Standardize the feature values
scaler = StandardScaler()
scaler.fit(X_train)
# Transform the feature values
X_train = scaler.transform(X_train)
# Train the model
model.fit(X_train, y_train)
def predict_label(raw_sample):
    # Preprocess the raw unprocessed data
    raw_sample = scaler.transform([raw_sample])  # Note: input is expected as 2D array
    # Predict and return the output
    return model.predict_proba(raw_sample)