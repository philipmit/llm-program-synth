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
# Normalize the datasets
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
# Train the model with logistic regression
lr_model = LogisticRegression(multi_class='ovr', solver='liblinear')
lr_model.fit(X_train_scaled, y_train)
def predict_label(raw_data):
    # The model expects scaled input data, so scale the raw_data input
    raw_data_reshaped = np.reshape(raw_data, (1, -1))
    scaled_data = scaler.transform(raw_data_reshaped)
    # Predict the probabilities using the logistic regression model
    probabilities = lr_model.predict_proba(scaled_data)
    return probabilities.flatten()