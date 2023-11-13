from sklearn.datasets import load_digits
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
import numpy as np
# Load the digits dataset
digits = load_digits()
X = digits.data
y = digits.target
# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)
# Train a Gradient Boosting Classifier
gb = GradientBoostingClassifier(learning_rate=0.01, n_estimators=500, max_depth=5, random_state=42)
gb.fit(X_train, y_train)
def predict_label(raw_data):
    # Reshape and normalize raw_data
    normalized_data = np.array(raw_data).reshape(1, -1) / 16.0
    # Make a prediction using the trained model
    pred_probs = gb.predict_proba(normalized_data)
    return pred_probs[0]