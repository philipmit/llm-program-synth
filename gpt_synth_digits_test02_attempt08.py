from sklearn.datasets import load_digits
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import numpy as np
# Load the digits dataset
digits = load_digits()
X = digits.data
y = digits.target
# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)
# Train a Random Forest Classifier with improved hyperparameters
rf = RandomForestClassifier(n_estimators=1000, max_depth=10, random_state=42)
rf.fit(X_train, y_train)
def predict_label(raw_data):
    # Reshape and normalize raw_data
    normalized_data = np.array(raw_data).reshape(1, -1) / 16.0
    # Make a prediction using the trained model
    pred_probs = rf.predict_proba(normalized_data)
    return pred_probs[0]