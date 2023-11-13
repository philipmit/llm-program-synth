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
# Normalize the data using StandardScaler which standardizes features by removing the mean and scaling to unit variance
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
# Train the Logistic Regression model 
log_reg = LogisticRegression(max_iter=10000)  # Increased max_iter to ensure convergence for this dataset
log_reg.fit(X_train, y_train)
# Define the predict_label function
def predict_label(raw_data):
    # Preprocess the raw data with the scaler
    raw_data = scaler.transform(raw_data.reshape(1, -1))
    # Predict the probabilities
    predicted_probabilities = log_reg.predict_proba(raw_data)
    return predicted_probabilities