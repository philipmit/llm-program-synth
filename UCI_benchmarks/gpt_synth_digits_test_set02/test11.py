from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
# Load the digits dataset
digits = load_digits()
X = digits.data
y = digits.target
# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)
# Initialize an instance of the standard scaler
scaler = StandardScaler()
# Fit it on the training data
scaler.fit(X_train)
# Transform the training data
X_train_scaled = scaler.transform(X_train)
# Set up logistic regression model
model = LogisticRegression(max_iter=1000)
# Train model
model.fit(X_train_scaled, y_train)
def predict_label(sample_data):
    global model
    global scaler
    # Preprocess the sample
    processed_sample = scaler.transform(sample_data.reshape(1, -1))
    # Use the model to predict the most likely class probabilities only
    predicted_probs = model.predict_proba(processed_sample)[0]
    return predicted_probs