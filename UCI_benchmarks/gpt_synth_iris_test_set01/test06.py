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
# Create a StandardScaler instance and fit it to the training data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
# Train the MLPClassifier model on the training data
mlp = MLPClassifier(hidden_layer_sizes=(10,), max_iter=1000)
mlp.fit(X_train_scaled, y_train)
# Define the predict_label function
def predict_label(raw_data):
    # Preprocess the raw data using the StandardScaler instance
    raw_data_scaled = scaler.transform(raw_data.reshape(1, -1))
    # Use the trained MLPClassifier model to predict the sample's label
    predicted_proabilities = mlp.predict_proba(raw_data_scaled)
    return predicted_proabilities