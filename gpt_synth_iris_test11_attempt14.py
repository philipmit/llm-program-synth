from sklearn.neural_network import MLPClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
# Load the Iris dataset
iris = load_iris()
X = iris.data
y = iris.target
# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Apply Standard Scaler for normalization
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
# Train the MLPClassifier
clf = MLPClassifier(hidden_layer_sizes=(10,), max_iter=1000, random_state=42)
clf.fit(X_train, y_train)
def predict_label(raw_sample):
    # Apply scaling to the raw_sample
    raw_sample_scaled = scaler.transform([raw_sample])
    # Remove the list wrapping from the return statement. 
    # Change from 'return clf.predict_proba(raw_sample_scaled)' to
    return clf.predict_proba(raw_sample_scaled)[0]