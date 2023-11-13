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
# Data normalization
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
# Define the MLPClassifier model
model = MLPClassifier(hidden_layer_sizes=(10,), random_state=42, max_iter=300)
# Train the model
model.fit(X_train, y_train)
def predict_label(raw_data):
    # Data normalization
    processed_data = scaler.transform([raw_data])
    # prediction
    predicted_probs = model.predict_proba(processed_data)
    return predicted_probs