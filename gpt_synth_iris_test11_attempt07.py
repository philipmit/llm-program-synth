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
# Initialize scaling object
sc = StandardScaler()
# Fit the scaler to training data
X_train = sc.fit_transform(X_train)
# Initialize MLPClassifier 
mlp = MLPClassifier(hidden_layer_sizes=(10, 10, 10), max_iter=1000)
# Train the model
mlp.fit(X_train, y_train)
def predict_label(raw_sample):
    # Preprocess the raw_sample with the same scaler
    transformed_sample = sc.transform([raw_sample])
    # Predict probabilities
    probabilities = mlp.predict_proba(transformed_sample)
    return probabilities[0]