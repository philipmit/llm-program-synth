import numpy as np
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
# Load wine dataset
wine = load_wine()
# Set data and target
X = wine.data
y = wine.target
# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)
# Standardize the data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
# Initialize and fit the logistic regression model
model = LogisticRegression(max_iter=2000)
model.fit(X_train, y_train)
# Defining the function to predict labels
def predict_label(raw_data):
    # Standardize the raw data
    standardized_data = scaler.transform(raw_data.reshape(1, -1))
    # Predict and return the class with the highest probability
    return np.argmax(model.predict_proba(standardized_data))