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
# Normalize the data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
# Train the logistic regression model
model = LogisticRegression(max_iter=10000)
model.fit(X_train, y_train)
def predict_label(X_sample):
    # Preprocess the input
    X_sample = scaler.transform([X_sample])
    # Predict and return the probabilities
    # The [0] at the end flattens the prediction from 3D to 2D  
    return model.predict_proba(X_sample)[0]