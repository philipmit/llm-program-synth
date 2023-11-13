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
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
# Define the multi-layer perceptron model
mlp = MLPClassifier(random_state=42, max_iter=300).fit(X_train_scaled, y_train)
def predict_label(raw_data):
    processed_data = scaler.transform(raw_data.reshape(1, -1)) # reshape the data and standardize it
    probabilities = mlp.predict_proba(processed_data) # get predicted probabilities for the sample
    return probabilities