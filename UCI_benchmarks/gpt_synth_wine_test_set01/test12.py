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
# Preprocess the data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
# Train the logistic regression model
model = LogisticRegression(random_state=42, max_iter=1000)
model.fit(X_train, y_train)
def predict_label(data):
    # Preprocess the example
    data = scaler.transform(data.reshape(1, -1))
    # Predict and return probabilities
    prediction = model.predict_proba(data)
    # Ensure that the output is a 1D array for single observation
    if prediction.shape[0] == 1:
        return prediction[0]
    return prediction