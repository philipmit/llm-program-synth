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
# Standardize the data (mean=0, std=1) using training data
scaler = StandardScaler().fit(X_train)
# Apply the standardization to both training and test sets
X_train_std = scaler.transform(X_train)
# Train the Logistic Regression model
model = LogisticRegression(random_state=42)
model.fit(X_train_std, y_train)
# Define the predict_label function
def predict_label(raw_data):
    # Apply the same standardization
    raw_data_std = scaler.transform([raw_data])
    # Predict probabilities
    prediction = model.predict_proba(raw_data_std)[0]
    return prediction