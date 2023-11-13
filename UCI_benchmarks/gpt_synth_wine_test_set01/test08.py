from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import numpy as np
# Load the wine dataset
wine = load_wine()
X = wine.data
y = wine.target
# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)
# Preprocess the training data
scaler = StandardScaler().fit(X_train)
# Transform the training data
X_train_std = scaler.transform(X_train)
# Create the logistic regression model
model = LogisticRegression(multi_class='ovr', solver='liblinear', random_state=42)
# Train the model
model.fit(X_train_std, y_train)
def predict_label(raw_data):
    # Preprocess the raw data
    raw_data = np.array(raw_data).reshape(1, -1)  # Reshape data if it's a single sample
    raw_data_std = scaler.transform(raw_data)
    # Predict the probabilities
    predicted_probabilities = model.predict_proba(raw_data_std)
    return predicted_probabilities[0]