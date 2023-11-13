from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
# Load the Iris dataset
iris = load_iris()
X = iris.data
y = iris.target
# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)
# Apply standard scaling to get optimized result
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
# Train the multi-layer perceptron
mlp = MLPClassifier(hidden_layer_sizes=(10, 10, 10), max_iter=1000)
mlp.fit(X_train, y_train)
def predict_label(single_sample_data):
    # The sample needs to be scaled using the scaler fitted on the training set
    single_sample_data_scaled = sc.transform([single_sample_data])
    # Return array of predicted probabilities
    return mlp.predict_proba(single_sample_data_scaled)[0]