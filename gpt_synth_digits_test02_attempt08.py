from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline
import numpy as np
# Load the digits dataset
digits = load_digits()
X = digits.data
y = digits.target
# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Making a pipeline for preprocessing and model
# Changing the classifier to RandomForest with more number of estimators
# and more depth to capture complex features.
pipeline = make_pipeline(
    StandardScaler(),
    PCA(n_components=40),
    RandomForestClassifier(n_estimators=500, max_depth=12, random_state=42))
# Train the model
pipeline.fit(X_train, y_train)
def predict_icu_mortality(raw_data):
    # Reshape the raw_data
    reshaped_data = np.array(raw_data).reshape(1, -1)
    # Make a prediction using the trained model
    pred_probs = pipeline.predict_proba(reshaped_data)
    return pred_probs[0]