from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
# Load iris data
iris = load_iris()
X = iris.data
y = iris.target
# Create train and test datasets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()  
X_train = scaler.fit_transform(X_train)
mlp = MLPClassifier(hidden_layer_sizes=(10, 10, 10), max_iter=1000)
mlp.fit(X_train, y_train)
def predict_label(sample_data):
    processed_sample = scaler.transform([sample_data])
    probabilities = mlp.predict_proba(processed_sample)[0] # Get the first element to ensure the result is 2D
    return probabilities