from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import numpy as np
# Load the digits dataset
digits = load_digits()
X = digits.data
y = digits.target
# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)
# Train the model using a RandomForestClassifier
# RandomForest model is chosen as it can handle a larger number of features and can prevent overfitting
rf = RandomForestClassifier(n_estimators=2000, max_depth=15, random_state=42)
rf.fit(X_train, y_train)
def predict_icu_mortality(raw_data):
    # Reshape the raw_data
    reshaped_data = np.array(raw_data).reshape(1, -1)
    # Normalize the reshaped_data
    normalized_data = reshaped_data / 16.0
    # Make a prediction using the trained model
    pred_probs = rf.predict_proba(normalized_data)
    return pred_probs[0]