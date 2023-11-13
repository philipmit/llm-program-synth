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
# Initialize StandardScaler
scaler = StandardScaler()
scaler.fit(X_train)
# Apply transform to both the training and testing sets
X_train_std = scaler.transform(X_train)
# Create an instance of MLPClassifier
mlp = MLPClassifier(hidden_layer_sizes=(10,10,10), max_iter=1000)
# Fit the model to data
mlp.fit(X_train_std, y_train)
# Function to predict the label
def predict_label(raw_data):
    # Apply the same transformation to the input data
    scaled_data = scaler.transform(raw_data.reshape(1, -1))
    # Use the trained model to predict the probability
    prob = mlp.predict_proba(scaled_data)
    # The prob variable is a 2D numpy array. The [0] is used to return a 1D list
    return prob[0]