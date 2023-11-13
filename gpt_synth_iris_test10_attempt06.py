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
# Create a StandardScaler and fit it to the training data
scaler = StandardScaler().fit(X_train)
# Transform the training data using the StandardScaler
X_train_scaled = scaler.transform(X_train)
# Create the MLPClassifier
mlp = MLPClassifier(hidden_layer_sizes=(10,), activation='relu', solver='adam', random_state=42)
# Train the MLPClassifier
mlp.fit(X_train_scaled, y_train)
def predict_label(raw_data):
    # Scale the raw_data using the previously fit StandardScaler
    processed_data = scaler.transform([raw_data])
    # Model's prediction probabilities
    predicted_proba = mlp.predict_proba(processed_data)
    return predicted_proba