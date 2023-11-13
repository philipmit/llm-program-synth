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
# Standardize the dataset 
scaler = StandardScaler()
scaler.fit(X_train)
# Transform the training data
X_train_scaled = scaler.transform(X_train)
# Train the MLPClassifier
model = MLPClassifier(hidden_layer_sizes=(10,10), max_iter=1000)
model.fit(X_train_scaled, y_train)
def predict_label(raw_data):
    # Standardize the raw data
    raw_data_scaled = scaler.transform([raw_data])
    # Use the trained model to predict the label
    proba = model.predict_proba(raw_data_scaled)
    return proba[0]