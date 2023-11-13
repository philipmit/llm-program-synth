from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV
# Load the Iris dataset
iris = load_iris()
X = iris.data
y = iris.target
# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Standardize the data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
# Create a Multi-Layer Perceptron Classifier with larger hidden layers
mlp = MLPClassifier(max_iter=1000)
# Use GridSearchCV to tune the model's parameters
param_grid = {
    'hidden_layer_sizes': [(50,50,50), (50,100,50), (100,)],
    'activation': ['tanh', 'relu'],
    'solver': ['sgd', 'adam'],
    'alpha': [0.0001, 0.05],
    'learning_rate': ['constant','adaptive'],
}
clf = GridSearchCV(mlp, param_grid, cv=5, scoring='roc_auc', verbose=2)
clf.fit(X_train_scaled, y_train)
# Update the scaler and MLPClassifier parameters
scaler = clf.best_estimator_.steps[0][1]
mlp = clf.best_estimator_.steps[1][1]
# Define the predict_label function
def predict_label(sample):
    standardized_sample = scaler.transform([sample])
    probabilities = mlp.predict_proba(standardized_sample)
    return probabilities.flatten()