from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import numpy as np
# Load the Breast Cancer dataset
breast_cancer = load_breast_cancer()
X = breast_cancer.data
y = breast_cancer.target
# Scale the data
scaler = StandardScaler()
X = scaler.fit_transform(X)
# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)
# Train the logistic regression model
model = LogisticRegression()
model.fit(X_train, y_train)
def predict_label(raw_data):
    """
    Predicts the probability of the label being 1 for raw, unprocessed data.
    Parameters:
    raw_data (np.array): Raw data for a single sample.
    Returns:
    float: Probability of the label being 1.
    """
    # Preprocess raw_data
    raw_data = scaler.transform([raw_data])
    # Predict probability
    proba = model.predict_proba(raw_data)
    return proba[0][1]