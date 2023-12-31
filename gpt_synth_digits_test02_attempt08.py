from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
import numpy as np
# Load the digits dataset
digits = load_digits()
X = digits.data
y = digits.target
# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)
# Build the pipeline with processing and logistic regression
pipeline = make_pipeline(
    StandardScaler(),
    PCA(n_components=30),
    LogisticRegression())
# Train the model
pipeline.fit(X_train, y_train)
# Define predict_label function 
def predict_label(raw_data):
    # Reshape the raw_data
    reshaped_data = np.array(raw_data).reshape(1, -1)
    # Make a prediction using the trained model
    pred_probs = pipeline.predict_proba(reshaped_data)
    return pred_probs[0]