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
X_train = scaler.fit_transform(X_train)
# Training the multilayer perceptron model
model = MLPClassifier(random_state=42)
model.fit(X_train, y_train)
# Defining the function for predicting labels
def predict_label(sample):
    # Reshaping sample to 2D array as it is required for the transform method
    sample = sample.reshape(1, -1)
    # Standardizing the sample
    sample = scaler.transform(sample)
    # Predicting probabilities
    predicted = model.predict_proba(sample)[0]  # Take the first element of predicted probabilities as it's a 2D array
    return predicted