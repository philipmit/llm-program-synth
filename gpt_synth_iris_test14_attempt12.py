from sklearn.neural_network import MLPClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
def predict_label(single_sample):
    single_sample = scaler.transform([single_sample])
    probabilities = model.predict_proba(single_sample)
    return probabilities
# Load the Iris dataset
iris = load_iris()
X = iris.data
y = iris.target
# Preprocessing
scaler = StandardScaler()
X = scaler.fit_transform(X)
# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)
# Define MLP classifier
model = MLPClassifier(hidden_layer_sizes=(100,), max_iter=500, random_state=42)
# Train the model
model.fit(X_train, y_train)