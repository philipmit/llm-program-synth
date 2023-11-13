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
# Standardize the data 
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
# Define the Multi-Layer Perceptron classifier and train
mlp = MLPClassifier(hidden_layer_sizes=(10, 10, 10), max_iter=1000)
mlp.fit(X_train_scaled, y_train)
# Save the scaler and the model to be used by the prediction function
scaler_model = scaler
mlp_model = mlp
# Define the predict_label function
def predict_label(raw_unprocessed_data):
    # Scale the input data
    scaled_data = scaler_model.transform(raw_unprocessed_data)
    # Perform prediction
    probabilities = mlp_model.predict_proba(scaled_data)
    return probabilities