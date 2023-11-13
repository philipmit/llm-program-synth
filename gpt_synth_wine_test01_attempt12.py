import numpy as np
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
# Load the wine dataset
wine = load_wine()
X = wine.data
y = wine.target
# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)
# Standardize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
# Train a logistic regression model
model = LogisticRegression(multi_class='ovr', solver='liblinear')
model.fit(X_train, y_train)
def predict_label(x_raw):
    x_raw = np.array(x_raw).reshape(1,-1)
    # Standardize the data
    x_std = scaler.transform(x_raw)
    # Use the model to predict probabilities
    proba = model.predict_proba(x_std)
    # Return the index of the max probability
    return np.argmax(proba, axis=1)