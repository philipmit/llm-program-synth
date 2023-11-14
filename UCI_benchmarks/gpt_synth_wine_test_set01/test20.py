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
# Data normalization
scaler = StandardScaler().fit(X_train)
# Apply normalization to training data
X_train_normalized = scaler.transform(X_train)
# Train the logistic regression model
lr = LogisticRegression(max_iter=10000)
lr.fit(X_train_normalized, y_train)
def predict_label(sample_data):
    # Ensure the sample data is 2D
    if len(sample_data.shape) == 1:
        sample_data = sample_data.reshape(1, -1)
    # Normalize the sample data
    sample_data_normalized = scaler.transform(sample_data)
    # Predict probabilities
    probabilities = lr.predict_proba(sample_data_normalized)
    return probabilities