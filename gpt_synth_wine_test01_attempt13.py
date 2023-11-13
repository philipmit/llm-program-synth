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
# Normalize the data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
# Train the logistic regression model
model = LogisticRegression()
model.fit(X_train_scaled, y_train)
def predict_label(raw_data):
    # Normalize the raw_data using the same scaler used in the training process
    raw_data_scaled = scaler.transform(raw_data.reshape(1, -1))
    # Get the predicted probabilities
    probs = model.predict_proba(raw_data_scaled)[0]  # Updated line here
    return probs