from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
# Load the Iris dataset
iris = load_iris()
X = iris.data
y = iris.target
# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)
# Instantiate the scaler and scale / normalize the training data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
# Define MLP classifier and train model
mlp = MLPClassifier(hidden_layer_sizes=(30,30,30))
mlp.fit(X_train_scaled,y_train)
# Define the predict_label function
def predict_label(raw_data):
    raw_data = raw_data.reshape(1, -1)
    scaled_data = scaler.transform(raw_data)
    return mlp.predict_proba(scaled_data)    