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
# Normalizing the features
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
# Train the MLPClassifier
clf = MLPClassifier(max_iter=1000, random_state=42)
clf.fit(X_train, y_train)
def predict_label(input_sample):
    # Transform the sample data
    input_sample = sc.transform(input_sample.reshape(1, -1))
    # Predict probabilities and return them
    probabilities = clf.predict_proba(input_sample)
    return probabilities[0]