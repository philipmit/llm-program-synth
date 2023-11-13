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
# Normalize dataset to have mean=0 and variance=1, this improves the performance and convergence of neural network
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
# Create neural network model using MLP (Multi-layer Perceptron)
mlp = MLPClassifier(hidden_layer_sizes=(10,), activation='relu', solver='adam', max_iter=300, random_state=42)
mlp.fit(X_train,y_train)  # Train the model
def predict_label(raw_sample):
    processed_sample = scaler.transform([raw_sample])  # Preprocess the sample, it must be a list of samples (even if we’ve just one)
    return mlp.predict_proba(processed_sample)[0]  # return the predicted probabilities