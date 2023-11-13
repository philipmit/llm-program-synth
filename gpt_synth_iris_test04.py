from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
# Load the Iris dataset
iris = load_iris()
X = iris.data
y = iris.target
# Split the dataset into a training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Necessity of standarization comes due to MLP's sensitivity to feature scaling
scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
# Create a MLP Classifier
mlp = MLPClassifier(hidden_layer_sizes=(10, 10), max_iter=1000)
# Train the model using the training sets
mlp.fit(X_train, y_train)
# Function to predict the label of the given raw unprocessed data
def predict_label(raw_data):
    processed_data = scaler.transform(raw_data.reshape(1, -1))  # Standarize and reshape raw_data
    probabilities = mlp.predict_proba(processed_data)[0]
    return probabilities