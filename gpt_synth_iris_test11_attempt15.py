import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
# Load the Iris dataset
iris = load_iris()
X = iris.data
y = iris.target
# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
# Preprocessing
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
# Define model 
model = MLPClassifier(hidden_layer_sizes=(100,), max_iter=300, activation = 'relu', solver='adam', random_state=42)
# Train the model
model.fit(X_train, y_train)
# Define predict_label function
def predict_label(raw_sample):
    # Preprocessing raw sample
    raw_sample_transformed = scaler.transform([raw_sample])
    # Predict
    prediction = np.argmax(model.predict_proba(raw_sample_transformed))
    return prediction