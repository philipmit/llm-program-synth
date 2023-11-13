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
# Standardize the features
scaler = StandardScaler()
scaler.fit(X_train)
# Apply the transformations to the data
X_train_scaled = scaler.transform(X_train)
# Initialize the Multi-Layer Perceptron Classifier
classifier = MLPClassifier(max_iter=1000)
# Train the model
classifier.fit(X_train_scaled, y_train)
def predict_label(sample):
    #apply same scaling to the sample
    sample_scaled = scaler.transform([sample])
    #predict using the trained model
    predicted_probabilities = classifier.predict_proba(sample_scaled)
    return predicted_probabilities