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
# Standardize the features
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
# Create an instance of the MLPClassifier
mlp = MLPClassifier(hidden_layer_sizes=(10,), max_iter=1000)
# Train the model
mlp.fit(X_train, y_train)
# Define the prediction function
def predict_label(raw_data):
    raw_data = sc.transform([raw_data])
    pred_proba = mlp.predict_proba(raw_data)
    return pred_proba[0]  # Select the first and only element of the list to keep it 2D