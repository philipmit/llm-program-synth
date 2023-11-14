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
# Standardize Features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
# Define the classifier
mlp = MLPClassifier(max_iter=1000)
# Train the model
mlp.fit(X_train, y_train)
def predict_label(raw_data):
    processed_data = scaler.transform([raw_data])  # note thar model was trained with standardized features
    return mlp.predict_proba(processed_data)