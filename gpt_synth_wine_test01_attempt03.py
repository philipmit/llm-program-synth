from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
# Load the wine dataset
wine = load_wine()
X = wine.data
y = wine.target
# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)
# Standardize the features to have zero mean and unit variance
scaler = StandardScaler().fit(X_train)
X_train = scaler.transform(X_train)
# Train a logistic regression model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)
def predict_label(raw_data):
    # Apply the same scaler to the raw_data
    scaled_data = scaler.transform([raw_data])
    # Use the trained model to predict probabilities
    probabilities = model.predict_proba(scaled_data)
    return probabilities[0]