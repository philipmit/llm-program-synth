from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
# Load the Iris dataset
iris = load_iris()
X = iris.data
y = iris.target
# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Create a standard scaler
scaler = StandardScaler()
# Create a Multi-Layer Perceptron Classifier
mlp = MLPClassifier(hidden_layer_sizes=(10,10,10), max_iter=1000)
# Create a pipeline to scale the data and then train the MLP Classifier
pipeline = make_pipeline(scaler, mlp)
# Train the model using the training data
pipeline.fit(X_train, y_train)
# Define a function to predict the labels based on the raw unprocessed data for a single sample
def predict_label(sample):
    # Standardize the data, since the model was trained with standardized data
    standardized_sample = scaler.transform([sample])
    probabilities = pipeline.predict_proba(standardized_sample)
    return probabilities.flatten()